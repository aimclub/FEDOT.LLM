from fedotllm.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig, FedotIndustrialConfig
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings 
import fedotllm.prompts as prompts

logger = get_logger()


def run_generate_automl_config(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running generate automl config")
    dataset_description = "\n".join(
        [
            (
                "<dataset-split>\n"
                + f"{split.name}\n"
                + "<features>\n"
                + split.get_description() + "\n"
                + "</features>\n"
                + "</dataset-split>"
            )
            for split in dataset.splits
        ]
    )
    match get_settings().config.automl.lower():
        case "fedot":
            config = inference.chat_completion(
                prompts.automl.generate_configuration_prompt(
                    reflection=state["reflection"],
                    dataset_description=dataset_description,
                ),
                structured=FedotConfig,
            )
        case "fedotind":
            config = inference.chat_completion(
                prompts.automl.generate_configuration_prompt(
                    reflection=state["reflection"],
                    dataset_description=dataset_description,
                ),
                structured=FedotIndustrialConfig,
            )

    state["fedot_config"] = config

    return state
