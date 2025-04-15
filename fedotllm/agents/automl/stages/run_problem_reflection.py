from langgraph.types import Command

import fedotllm.prompting.prompts as prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm import LiteLLMModel
from fedotllm.log import get_logger
from fedotllm.tabular import Dataset

logger = get_logger()


def run_problem_reflection(
    state: AutoMLAgentState, llm: LiteLLMModel, dataset: Dataset
):
    logger.info("Running problem reflection")

    reflection = llm.query(
        prompts.automl.problem_reflection_prompt(
            user_description=state["description"],
            data_files_and_content=dataset.dataset_preview(),
        )
    )
    return Command(update={"reflection": reflection})
