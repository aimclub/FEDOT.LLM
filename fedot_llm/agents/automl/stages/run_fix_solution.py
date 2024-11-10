from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.llm.inference import AIInference
from settings.config_loader import get_settings
from fedot_llm.agents.utils import extract_code
from fedot_llm.log import get_logger

logger = get_logger()


def run_fix_solution(state: AutoMLAgentState, inference: AIInference):
    logger.info("Running fix solution")
    solution = state['solutions'][-1]
    fix_prompt = get_settings().get("prompts.automl.run_fix_solution.user").format(description=state['description'],
                                                                                   code_recent_solution=solution['code'],
                                                                                   trace=solution['exec_result'].sandbox_result)
    fixed_solution = inference.chat_completion(fix_prompt)
    extracted_code = extract_code(fixed_solution)
    if extracted_code:
        solution['code'] = extracted_code
        solution['fix_tries'] += 1
    else:
        raise ValueError("No code found in the response")
    return state
