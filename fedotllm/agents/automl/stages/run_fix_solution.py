from fedotllm.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.utils import extract_code
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
import fedotllm.prompts as prompts

import os

logger = get_logger()


def run_fix_solution(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running fix solution")
    codegen = state["codegen_sol"]
    solution = state["solutions"][-1]
    exec_result = solution["exec_result"]
    files = "\n".join(
        [
            f"File: {file.name}\n"
            + "\n".join([f"- {col}" for col in file.data.columns])
            for file in dataset.splits
        ]
    )
    dataset_path = os.path.relpath(dataset.path, dataset.path.cwd())
    fix_prompt = prompts.automl.fix_solution_prompt(
        user_instruction=state["description"],
        dataset_path=dataset_path,
        files=files,
        code_recent_solution=codegen["code"],
        stderr=exec_result.stderr,
        stdout=exec_result.stdout,
    )
    fixed_solution = inference.chat_completion(fix_prompt)
    extracted_code = extract_code(fixed_solution)
    if extracted_code:
        codegen["code"] = extracted_code
        codegen["fix_tries"] += 1
    else:
        raise ValueError("No code found in the response")
    return state
