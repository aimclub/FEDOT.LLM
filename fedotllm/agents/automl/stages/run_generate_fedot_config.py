from fedotllm.agents.automl.data.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


def run_generate_fedot_config(state: AutoMLAgentState, inference: AIInference, dataset: Dataset):
    logger.info("Running generate fedot config")
    dataset_description = "\n".join([
        (
                "<dataset-split>\n" +
                f"{split.name}\n" +
                "<features>\n" +
                '\n'.join([f'- {col}' for col in split.data.columns]) +
                "</features>\n" +
                "</dataset-split>"
        )
        for split in dataset.splits
    ])
    fedot_config = inference.chat_completion(
        get_settings().prompts.automl.run_generate_fedot_config.user.format(
            reflection=state['reflection'], dataset_description=dataset_description),
        structured=FedotConfig
    )
    state['fedot_config'] = fedot_config

    return state
