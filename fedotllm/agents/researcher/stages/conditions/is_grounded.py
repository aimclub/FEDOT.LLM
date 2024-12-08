from agents.researcher.structured import GradeHallucination, BoolAnswer
from agents.utils import render
from settings.config_loader import get_settings
from agents.researcher.state import ResearcherAgentState
from llm.inference import AIInference


def is_grounded(state: ResearcherAgentState, inference: AIInference):
    grade = GradeHallucination.model_validate(inference.chat_completion(*render(get_settings().get("prompts.researcher.is_grounded"),
                                                                                generation=state["generation"], documents=state["documents"]), structured=GradeHallucination))
    return grade.score == BoolAnswer.YES
