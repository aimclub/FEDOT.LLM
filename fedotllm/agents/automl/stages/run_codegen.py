from fedotllm.data.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState, Solution
from fedotllm.agents.utils import extract_code
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
import fedotllm.prompts as prompts

import os

logger = get_logger()


def run_codegen(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running codegen")
    files = "\n".join(
        [
            f"File: {file.name}\n"
            + "\n".join([f"- {col}" for col in file.data.columns])
            for file in dataset.splits
        ]
    )
    dataset_path = os.path.relpath(dataset.path, dataset.path.cwd())
    codegen_prompt = prompts.automl.code_generation_prompt(
        user_instruction=state["description"],
        skeleton=state["skeleton"],
        dataset_path=dataset_path,
        files=files,
    )
    code = inference.chat_completion(codegen_prompt)
    extracted_code = extract_code(code)
    if extracted_code:
        state["codegen_sol"] = Solution(
            code=extracted_code, exec_result=None, fix_tries=0
        )
    else:
        raise ValueError("No code found in the response")
    return state
