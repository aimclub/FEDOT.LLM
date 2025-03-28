from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GradeAnswer, BoolAnswer
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


def is_useful(state: ResearcherAgentState, inference: AIInference):
    grade = GradeAnswer.model_validate(
        inference.chat_completion(
            prompts.researcher.is_useful_prompt(state["generation"], state["question"]),
            structured=GradeAnswer,
        )
    )
    return grade.score == BoolAnswer.YES
