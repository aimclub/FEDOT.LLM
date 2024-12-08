from fedot_llm.agents.automl.structured import ProblemReflection
from fedot_llm.agents.utils import render
from fedot_llm.llm.inference import AIInference
from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.log import get_logger

logger = get_logger()

def run_problem_reflection(state: AutoMLAgentState, inference: AIInference):
    logger.info("Running problem reflection")
    reflection = inference.chat_completion(
        *render(prompt='run_problem_reflection', format_vals=state),
        structured=ProblemReflection
    )
    state['reflection'] = reflection
    return state