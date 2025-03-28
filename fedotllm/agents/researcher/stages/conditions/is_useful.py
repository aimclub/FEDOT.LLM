from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GradeAnswer, BoolAnswer
from fedotllm.agents.utils import render
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def is_useful(state: ResearcherAgentState, inference: AIInference):
    grade = GradeAnswer.model_validate(
        inference.chat_completion(
            *render(
                get_settings().get("prompts.researcher.is_useful"),
                generation=state["generation"],
                question=state["question"],
            ),
            structured=GradeAnswer,
        )
    )
    return grade.score == BoolAnswer.YES
