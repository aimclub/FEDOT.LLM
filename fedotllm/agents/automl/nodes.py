import re
from pathlib import Path

from autoflake import fix_code
from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure
from langgraph.types import Command

from fedotllm import prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig, ProblemReflection
from fedotllm.agents.automl.templates.load_template import (
    load_template,
    render_template,
)
from fedotllm.agents.utils import extract_code
from fedotllm.data import Dataset
from fedotllm.enviroments.simple_eval import (
    execute_code,
)
from fedotllm.llm import AIInference
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


def problem_reflection(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running problem reflection")
    dataset_description = "\n".join(
        [
            (
                "<dataset-split>\n"
                + f"{split.name}\n"
                + "<features>\n"
                + "\n".join([f"- {col}" for col in split.data.columns])
                + "</features>\n"
                + "</dataset-split>"
            )
            for split in dataset.splits
        ]
    )

    reflection = inference.create(
        prompts.automl.problem_reflection_prompt(
            description=state["description"], dataset_description=dataset_description
        ),
        response_model=ProblemReflection,
    )
    state["reflection"] = reflection
    return Command(update={"reflection": reflection})


def generate_automl_config(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running generate automl config")
    dataset_description = "\n".join(
        [
            (
                "<dataset-split>\n"
                + f"{split.name}\n"
                + "<features>\n"
                + "\n".join([f"- {col}" for col in split.data.columns])
                + "</features>\n"
                + "</dataset-split>"
            )
            for split in dataset.splits
        ]
    )

    config = inference.create(
        prompts.automl.generate_configuration_prompt(
            reflection=state["reflection"],
            dataset_description=dataset_description,
        ),
        response_model=FedotConfig,
    )

    return Command(update={"fedot_config": config})


def select_skeleton(state: AutoMLAgentState, dataset: Dataset, workspace: Path) -> AutoMLAgentState:
    logger.info("Running select skeleton")
    fedot_config = state["fedot_config"]

    # Get prediction method
    predict_method = {
        "predict": "predict(features=input_data)",
        "forecast": "forecast(pre_history=input_data)",
        "predict_proba": "predict_proba(features=input_data)",
    }.get(fedot_config.predict_method)

    if predict_method is None:
        raise ValueError(f"Unknown predict method: {fedot_config.predict_method}")

    skeleton = load_template("skeleton")
    skeleton = render_template(
        template=skeleton,
        dataset_path=dataset.path,
        work_dir_path=workspace.resolve(),
    )

    return Command(update={"skeleton": skeleton})


def generate_code(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Generating code")
    files = "\n".join(
        [
            f"File: {file.name}\n"
            + "\n".join([f"- {col}" for col in file.data.columns])
            for file in dataset.splits
        ]
    )
    codegen_prompt = prompts.automl.code_generation_prompt(
        user_instruction=state["description"],
        skeleton=state["skeleton"],
        dataset_path=str(dataset.path.absolute()),
        files=files,
    )
    code = inference.query(codegen_prompt)
    extracted_code = extract_code(code)
    return Command(update={"raw_code": extracted_code})


def insert_templates(state: AutoMLAgentState):
    logger.info("Running insert templates")
    code = state["raw_code"]
    fedot_config = state["fedot_config"]
    predict_method = {
        "predict": "predict(features=input_data)",
        "forecast": "forecast(pre_history=input_data)",
        "predict_proba": "predict_proba(features=input_data)",
    }.get(fedot_config.predict_method)
    try:
        fedot_train = load_template("fedot_train.py")
        fedot_train = render_template(
            template=fedot_train,
            problem=f"{fedot_config.problem}",
            timeout=fedot_config.timeout,
            cv_folds=fedot_config.cv_folds,
            preset=f"'{fedot_config.preset.value}'",
            metric=f"'{fedot_config.metric.value}'",
        )
        fedot_evaluate = load_template("fedot_evaluate.py")
        fedot_evaluate = render_template(
            template=fedot_evaluate,
            problem=f"{fedot_config.problem}",
            predict_method=predict_method,
        )
        fedot_predict = load_template("fedot_predict.py")
        fedot_predict = render_template(
            template=fedot_predict,
            problem=f"{fedot_config.problem}",
            predict_method=predict_method,
        )
        automl_temp = "\n".join([fedot_train, fedot_evaluate, fedot_predict])
        code = code.replace(
            "from automl import train_model, evaluate_model, automl_predict",
            automl_temp,
        )
    except Exception:
        logger.error("Model removed template anchors")
        return Command(update={"code": None})

    code = fix_code(code, remove_all_unused_imports=True, remove_unused_variables=True)
    logger.debug(f"Updated code: \n{code}")
    return Command(update={"code": code})


def _generate_code_file(code: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "solution.py", "w") as f:
        f.write(code)
    return output_dir / "solution.py"


def evaluate(state: AutoMLAgentState, workspace: Path):
    logger.info("Running evaluate")
    code_path = _generate_code_file(state["code"], workspace)
    observation = execute_code(path_to_run_code=code_path)
    if observation.error:
        logger.error(observation.stderr)
    logger.debug(
        f"Evaluate result\nIs Error: {observation.error}\nStdout: {observation.stdout}\nStderr: {observation.stderr}"
    )
    return Command(update={"code_observation": observation})


def if_bug(state: AutoMLAgentState):
    if (
        state["code_observation"].error
        and state["fix_attempts"] <= get_settings().config.fix_tries
    ):
        return True
    if state["fix_attempts"] > get_settings().config.fix_tries:
        logger.error("Too many fix tries")
    return False


def fix_solution(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running fix solution")
    files = "\n".join(
        [
            f"File: {file.name}\n"
            + "\n".join([f"- {col}" for col in file.data.columns])
            for file in dataset.splits
        ]
    )

    fix_prompt = prompts.automl.fix_solution_prompt(
        user_instruction=state["description"],
        dataset_path=str(dataset.path.absolute()),
        files=files,
        code_recent_solution=state["raw_code"],
        stderr=state["code_observation"].stderr,
        stdout=state["code_observation"].stdout,
    )

    fixed_solution = inference.query(fix_prompt)
    extracted_code = extract_code(fixed_solution)
    return Command(
        update={"raw_code": extracted_code, "fix_attempts": state["fix_attempts"] + 1}
    )


def _extract_metrics(raw_output: str):
    pattern = r"Model metrics:\s*(\{.*?\})"
    match = re.search(pattern, raw_output)
    if match:
        return match.group(1).strip()
    return None


def extract_metrics(state: AutoMLAgentState, workspace: Path):
    logger.info("Running extract_metrics")
    state["metrics"] = _extract_metrics(state["code_observation"].stdout)
    logger.info(f"Metrics: {state['metrics']}")

    pipeline_path = workspace / "pipeline"
    if Path(pipeline_path).exists():
        model = Fedot(problem="classification")
        model.load(pipeline_path)
        state["pipeline"] = graph_structure(model.current_pipeline)
        logger.info(f"Pipeline: {state['pipeline']}")
    else:
        logger.error("Pipeline not found")
    return state


def generate_report(state: AutoMLAgentState, inference: AIInference):
    if state["code"] and state["pipeline"]:
        response = inference.query(
            prompts.automl.reporter_prompt(
                description=state["description"],
                metrics=state["metrics"],
                pipeline=state["pipeline"],
                code=state["code"],
            )
        )
    else:
        response = "Solution not found. Please try again."
    state["report"] = response
    return state
