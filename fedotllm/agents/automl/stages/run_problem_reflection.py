from fedotllm.data import Dataset
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import ProblemReflection
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
import fedotllm.prompts as prompts

logger = get_logger()


def run_problem_reflection(
    state: AutoMLAgentState, inference: AIInference, dataset: Dataset
):
    logger.info("Running problem reflection")
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

    reflection = inference.chat_completion(
        prompts.automl.problem_reflection_prompt(
            description=state["description"], dataset_description=dataset_description
        ),
        structured=ProblemReflection,
    )
    state["reflection"] = reflection
    return state
