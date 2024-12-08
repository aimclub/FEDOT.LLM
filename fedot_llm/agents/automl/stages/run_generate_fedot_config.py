from fedot_llm.agents.automl.structured import FedotConfig
from fedot_llm.agents.utils import render
from fedot_llm.llm.inference import AIInference
from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.log import get_logger

logger = get_logger()

def run_generate_fedot_config(state: AutoMLAgentState, inference: AIInference ):
    logger.info("Running generate fedot config")
    fedot_config = inference.chat_completion(
        *render(prompt='run_generate_fedot_config', format_vals=state),
        structured=FedotConfig
    )
    state['fedot_config'] = fedot_config
    
    return state
    