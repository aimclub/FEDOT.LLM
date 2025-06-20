import re
from pathlib import Path

import pandas as pd
from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure
from langchain_core.messages import HumanMessage, convert_to_openai_messages
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
from fedotllm.enviroments import (
    Observation,
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

    messages = convert_to_openai_messages(state["messages"])
    messages.append(
        {
            "role": "user",
            "content": prompts.automl.problem_reflection_prompt(
                data_files_and_content=dataset.dataset_preview(),
                dataset_eda=dataset.dataset_eda(),
            ),
        }
    )
    reflection = inference.query(messages)
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
                    "timeout": 1.0,
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
    return Command(update={"observation": observation})


def if_bug(state: AutoMLAgentState):
    if (
        state["observation"].error
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
        msg=state["observation"].msg,
        stderr=state["observation"].stderr,
        stdout=state["observation"].stdout,
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
        state["metrics"] = _parse_metrics(state["observation"].stdout)
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


def run_tests(state: AutoMLAgentState, workspace: Path, inference: AIInference):
    logger.info("Running tests")

    def extract_metrics(raw_output: str) -> Observation:
        if match := re.search(r"Model metrics:\s*(\{.*?\})", raw_output):
            return Observation(error=False, msg=match.group(1).strip())
        return Observation(
            error=True,
            msg="Metrics not found. Check if you use `evaluate` function and it was executed successfully.",
        )

    def extract_pipeline(workspace: Path) -> Observation:
        pipeline_path = workspace / "pipeline"
        if not pipeline_path.exists():
            return Observation(
                error=True,
                msg="Pipeline not found. Check if you use `train_model` function and it was executed successfully.",
            )
        try:
            model = Fedot(problem="classification")
            model.load(pipeline_path)
            return Observation(error=False, msg=graph_structure(model.current_pipeline))
        except Exception as e:
            return Observation(error=True, msg=f"Pipeline loading failed: {str(e)}")

    def check_submission_file(workspace: Path) -> Observation:
        submission_file = workspace / "submission.csv"
        return Observation(
            error=not submission_file.exists(),
            msg="Submission file exists."
            if submission_file.exists()
            else f"Submission file not found. Check if you save submission file successfully to {submission_file}.",
        )

    def test_submission_format(args: tuple) -> Observation:
        raw_output, inference = args
        submission_file = workspace / "submission.csv"
        print("DEBUG: RAW OUTPUT\n", raw_output)

        if not (
            match := re.search(
                r"Sample Submission File:\s*(.*?)$", raw_output, re.MULTILINE
            )
        ):
            return Observation(
                error=True,
                msg="Sample submission file format not found. Print `Sample Submission File: {sample_submission}` in your code so I can check it.",
            )

        sample_path = match.group(1).strip()
        print(f"Sample submission file path: {sample_path}")
        if not sample_path.endswith(".csv"):
            return Observation(
                error=True,
                msg="Sample Submission file format is incorrect. It should be a CSV file (.csv).",
            )

        if not submission_file.exists() or submission_file.suffix != ".csv":
            return Observation(
                error=True,
                msg="Submission file format is incorrect. It should be a CSV file (.csv).",
            )

        try:
            sample_df = pd.read_csv(sample_path)
            submission_df = pd.read_csv(submission_file)

            if submission_df.empty:
                return Observation(error=True, msg="Submission file is empty.")

            if not submission_df.columns.equals(sample_df.columns):
                return Observation(
                    error=True,
                    msg=f"Submission file columns don't match. Expected: {list(sample_df.columns)}, Got: {list(submission_df.columns)}",
                )

            if submission_df.shape[1] != sample_df.shape[1]:
                return Observation(
                    error=True,
                    msg=f"Submission file has wrong number of columns. Expected: {sample_df.shape[1]}, Got: {submission_df.shape[1]}",
                )

            # LLM validation for deeper format checking
            try:
                submission_sample = submission_df.head(3).to_string(
                    max_rows=3, max_cols=10
                )
                sample_submission_sample = sample_df.head(3).to_string(
                    max_rows=3, max_cols=10
                )

                result = inference.query(
                    prompts.utils.ai_assert_prompt(
                        var1=submission_sample,
                        var2=sample_submission_sample,
                        condition=(
                            "Compare the submission file format with the sample submission file format to determine if they have the same structure by verifying the following:"
                            "1. Column names match exactly. 2. Data types in corresponding columns are compatible. 3. The overall structure, including column order and presence, is consistent.\n"
                            "Note: Ignore differences in the values within the columns. Focus solely on structure, column names, and data types."
                        ),
                    )
                )

                if result.strip().lower() != "true":
                    return Observation(
                        error=True,
                        msg=f"Submission file format does not match expected format. Expected: {sample_submission_sample}, Got: {submission_sample}",
                    )
            except Exception:
                pass  # LLM validation is optional, pandas validation is sufficient

            return Observation(error=False, msg="Submission file format is correct.")

        except Exception as e:
            return Observation(
                error=True, msg=f"Error validating submission format: {str(e)}"
            )

    # Run all tests
    tests = [
        (extract_metrics, state["observation"].stdout),
        (extract_pipeline, workspace),
        (check_submission_file, workspace),
        (test_submission_format, (state["observation"].stdout, inference)),
    ]

    for test_func, param in tests:
        result = test_func(param)
        if result.error:
            logger.error(f"Test failed: {result.msg}")
            state["observation"].error = True
            state["observation"].msg += f"\nTest failed: {result.msg}"
        else:
            logger.info(f"Test passed: {result.msg}")
            state["observation"].msg += f"\nTest passed: {result.msg}"

    return state


def generate_report(state: AutoMLAgentState, inference: AIInference):
    if state["code"] and state["pipeline"]:
        messages = state["messages"]
        messages.append(
            {
                "role": "user",
                "content": prompts.automl.reporter_prompt(
                    metrics=state["metrics"],
                    pipeline=state["pipeline"],
                    code=state["code"],
                ),
            }
        )
        response = inference.query(convert_to_openai_messages(messages))
    else:
        response = "Solution not found. Please try again."
    return Command(
        update={"messages": HumanMessage(content=response, role="AutoMLAgent")}
    )
