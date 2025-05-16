import re
from pathlib import Path

from autoflake import fix_code
from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure
from langgraph.types import Command

from fedotllm import prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig
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
from fedotllm.log import logger
from fedotllm.settings.config_loader import get_settings

PREDICT_METHOD_MAP = {
    "predict": "predict(features=input_data)",
    "forecast": "forecast(pre_history=input_data)",
    "predict_proba": "predict_proba(features=input_data)",
}


def problem_reflection(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running problem reflection")

    reflection = inference.query(
        prompts.automl.problem_reflection_prompt(
            user_description=state["description"],
            data_files_and_content=dataset.dataset_preview(),
            dataset_eda=dataset.dataset_eda(),
        )
    )
    return Command(update={"reflection": reflection})


def generate_automl_config(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running generate automl config")

    config = inference.create(
        prompts.automl.generate_configuration_prompt(
            reflection=state["reflection"],
        ),
        response_model=FedotConfig,
    )

    return Command(update={"fedot_config": config})


def select_skeleton(state: AutoMLAgentState, dataset: Dataset, workspace: Path):
    logger.info("Running select skeleton")
    fedot_config = state["fedot_config"]

    # Get prediction method
    predict_method = PREDICT_METHOD_MAP.get(fedot_config.predict_method)

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
    codegen_prompt = prompts.automl.code_generation_prompt(
        reflection=state["reflection"],
        skeleton=state["skeleton"],
        dataset_path=str(dataset.path.absolute()),
    )
    code = inference.query(codegen_prompt)
    extracted_code = extract_code(code)
    return Command(update={"raw_code": extracted_code})


def insert_templates(state: AutoMLAgentState):
    logger.info("Running insert templates")
    code = state["raw_code"]
    fedot_config = state["fedot_config"]
    predict_method = PREDICT_METHOD_MAP.get(fedot_config.predict_method)

    try:
        templates = {
            "fedot_train.py": {
                "params": {
                    "problem": str(fedot_config.problem),
                    "timeout": fedot_config.timeout,
                    "cv_folds": fedot_config.cv_folds,
                    "preset": f"'{fedot_config.preset.value}'",
                    "metric": f"'{fedot_config.metric.value}'",
                }
            },
            "fedot_evaluate.py": {
                "params": {
                    "problem": str(fedot_config.problem),
                    "predict_method": predict_method,
                }
            },
            "fedot_predict.py": {
                "params": {
                    "problem": str(fedot_config.problem),
                    "predict_method": predict_method,
                }
            },
        }

        rendered_templates = []
        for template_name, config in templates.items():
            template = load_template(template_name)
            rendered = render_template(template=template, **config["params"])
            rendered_templates.append(rendered)

        code = code.replace(
            "from automl import train_model, evaluate_model, automl_predict",
            "\n".join(rendered_templates),
        )
        code = fix_code(
            code, remove_all_unused_imports=True, remove_unused_variables=True
        )
        logger.debug(f"Updated code: \n{code}")
        return Command(update={"code": code})

    except Exception as e:
        logger.error(f"Failed to insert templates: {str(e)}")
        return Command(update={"code": None})


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
        and state["fix_attempts"] < get_settings().config.fix_tries
    ):
        return True
    if state["fix_attempts"] >= get_settings().config.fix_tries:
        logger.error("Too many fix tries")
    return False


def fix_solution(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running fix solution")

    fix_prompt = prompts.automl.fix_solution_prompt(
        reflection=state["reflection"],
        dataset_path=str(dataset.path.absolute()),
        code_recent_solution=state["raw_code"],
        stderr=state["code_observation"].stderr,
        stdout=state["code_observation"].stdout,
    )

    fixed_solution = inference.query(fix_prompt)
    extracted_code = extract_code(fixed_solution)
    return Command(
        update={"raw_code": extracted_code, "fix_attempts": state["fix_attempts"] + 1}
    )


def extract_metrics(state: AutoMLAgentState, workspace: Path):
    logger.info("Running extract_metrics")

    def _parse_metrics(raw_output: str) -> str | None:
        pattern = r"Model metrics:\s*(\{.*?\})"
        if match := re.search(pattern, raw_output):
            return match.group(1).strip()
        return "Metrics not found"

    try:
        state["metrics"] = _parse_metrics(state["code_observation"].stdout)
        logger.info(f"Metrics: {state['metrics']}")

        pipeline_path = workspace / "pipeline"
        if pipeline_path.exists():
            model = Fedot(problem="classification")
            model.load(pipeline_path)
            state["pipeline"] = graph_structure(model.current_pipeline)
            logger.info(f"Pipeline: {state['pipeline']}")
        else:
            logger.warning("Pipeline not found at expected path")
            state["pipeline"] = "Pipeline not found"
    except Exception as e:
        logger.error(f"Failed to extract metrics: {str(e)}")
        state["metrics"] = "Metrics not found"
        state["pipeline"] = "Pipeline not found"

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
