from fedotllm.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig, FedotIndustrialConfig
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


def run_generate_automl_config(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running generate automl config")
    match get_settings().config.automl.lower():
        case 'fedot':
            config = inference.chat_completion(
                get_settings().prompts.automl.run_generate_automl_config.user.format(
                    reflection=state['reflection'], 
                    dataset_description=state['dataset_splits_description']),
                structured=FedotConfig
            )
        case 'fedotind':
            config = inference.chat_completion(
                get_settings().prompts.automl.run_generate_automl_config.user.format(
                    reflection=state['reflection'], 
                    dataset_description=state['dataset_splits_description']),
                structured=FedotIndustrialConfig
            )
            
    state['fedot_config'] = config

    return state
