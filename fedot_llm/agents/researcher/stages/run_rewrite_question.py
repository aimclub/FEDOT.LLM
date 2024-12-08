from fedot_llm.agents.researcher.state import ResearcherAgentState
from fedot_llm.agents.researcher.structured import RewriteQuestion
from fedot_llm.agents.utils import render
from fedot_llm.llm.inference import AIInference
from settings.config_loader import get_settings


def run_rewrite_question(state: ResearcherAgentState, inference: AIInference) -> ResearcherAgentState:
    state['question'] = RewriteQuestion.model_validate(inference.chat_completion(*render(get_settings.get("prompts.researcher.rewrite_question"),
                                                                                          question=state['question']),
                                                                                 structured=RewriteQuestion)).question
    return state
