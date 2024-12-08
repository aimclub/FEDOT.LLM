from agents.researcher.structured import GradeAnswer, BoolAnswer
from agents.utils import render
from settings.config_loader import get_settings
from agents.researcher.state import ResearcherAgentState
from llm.inference import AIInference


def is_useful(state: ResearcherAgentState, inference: AIInference):
    grade = GradeAnswer.model_validate(inference.chat_completion(*render(get_settings().get("prompts.researcher.is_useful"),
                                                                         generation=state["generation"], question=state["question"]), structured=GradeAnswer))
    return grade.score == BoolAnswer.YES
