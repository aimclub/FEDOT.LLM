import fedotllm.prompting.prompts as prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.llm import LiteLLMModel
from fedotllm.log import get_logger
from fedotllm.tabular import Dataset

logger = get_logger()


def run_generate_automl_config(
    state: AutoMLAgentState, llm: LiteLLMModel, dataset: Dataset
):
    logger.info("Running generate automl config")
    dataset_description = "\n".join(
        [
            (
                "<dataset-split>\n"
                + f"{split.name}\n"
                + "<features>\n"
                + "\n".join([f"- {col}" for col in split.data.columns])
                + "</features>\n"
                + "</dataset-split>"
            )
            for split in dataset.splits
        ]
    )

    config = llm.create(
        messages=[
            {
                "role": "user",
                "content": prompts.automl.generate_configuration_prompt(
                    reflection=state["reflection"],
                    dataset_description=dataset_description,
                ),
            }
        ],
        response_model=FedotConfig,
    )

    state["fedot_config"] = config

    return state
