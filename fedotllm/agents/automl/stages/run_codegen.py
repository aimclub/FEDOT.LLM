import fedotllm.prompting.prompts as prompts
from fedotllm.agents.automl.state import AutoMLAgentState, Solution
from fedotllm.llm import LiteLLMModel
from fedotllm.log import get_logger
from fedotllm.tabular.data import Dataset
from fedotllm.utils.parsers import extract_code

logger = get_logger()


def run_codegen(state: AutoMLAgentState, llm: LiteLLMModel, dataset: Dataset):
    logger.info("Running codegen")
    assert state["skeleton"] is not None, "Skeleton is not found"
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
        dataset_path=str(dataset.path.relative_to(dataset.path.cwd())),
        files=files,
    )
    code = llm.query(codegen_prompt)
    extracted_code = extract_code(code)
    if extracted_code:
        state["codegen_sol"] = Solution(
            code=extracted_code, exec_result=None, fix_tries=0
        )
    else:
        raise ValueError("No code found in the response")
    return state
