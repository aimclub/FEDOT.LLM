from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.llm.inference import AIInference
from settings.config_loader import get_settings
from fedot_llm.agents.utils import extract_code
from fedot_llm.log import get_logger

logger = get_logger()

def run_codegen(state: AutoMLAgentState, inference: AIInference):
    logger.info("Running codegen")
    codegen_prompt = get_settings().get("prompts.automl.run_codegen.user").format(user_instruction=state['description'], skeleton=state['skeleton'])
    code = inference.chat_completion(codegen_prompt)
    extracted_code = extract_code(code)
    if extracted_code:
        if "solutions" not in state:
            state['solutions'] = []
        state['solutions'].append({
            'code': extracted_code,
            'exec_result': None,
            'fix_tries': 0
        })
    else:
        raise ValueError("No code found in the response")
    return state
