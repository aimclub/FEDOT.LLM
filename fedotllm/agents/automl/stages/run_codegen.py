from agents.automl.state import AutoMLAgentState
from agents.automl.llm.inference import AIInference
from settings.config_loader import get_settings
from agents.automl.utils import extract_code
from log import get_logger
from agents.automl.data.data import Dataset

logger = get_logger()


def run_codegen(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running codegen")
    files = "\n".join([
        f"File: {file.name}\n" +
            "\n".join([f"- {col}" for col in file.data.columns])
        for file in dataset.splits
    ])
    codegen_prompt = get_settings().prompts.automl.run_codegen.user.format(user_instruction=state['description'],
                                                                           skeleton=state['skeleton'],
                                                                           dataset_path=dataset.path.relative_to(
                                                                               dataset.path.cwd()),
                                                                           files=files)
    code = inference.chat_completion(codegen_prompt)
    extracted_code = extract_code(code)
    if extracted_code:
        if "codegen_sol" not in state:
            state['codegen_sol'] = []
        state['codegen_sol'] = {
            'code': extracted_code,
            'exec_result': None,
            'fix_tries': 0
        }
    else:
        raise ValueError("No code found in the response")
    return state
