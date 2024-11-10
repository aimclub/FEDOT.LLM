from fedot_llm.agents.researcher.state import ResearcherAgentState
from fedot_llm.llm.inference import AIInference
from fedot_llm.agents.utils import render
from settings.config_loader import get_settings
from fedot_llm.agents.researcher.structured import GradeHallucination, BoolAnswer
def is_grounded(state: ResearcherAgentState, inference: AIInference):
    grade = GradeHallucination.model_validate(inference.chat_completion(*render(get_settings().get("prompts.researcher.is_grounded"),
                                               generation=state["generation"], documents=state["documents"]), structured=GradeHallucination))
    return grade.score == BoolAnswer.YES