from settings.config_loader import get_settings
from agents.researcher.state import ResearcherAgentState
from agents.researcher.structured import RewriteQuestion
from agents.utils import render
from llm.inference import AIInference


def run_rewrite_question(state: ResearcherAgentState, inference: AIInference) -> ResearcherAgentState:
    state['question'] = RewriteQuestion.model_validate(inference.chat_completion(*render(get_settings.get("prompts.researcher.rewrite_question"),
                                                                                         question=state['question']),
                                                                                 structured=RewriteQuestion)).question
    return state
