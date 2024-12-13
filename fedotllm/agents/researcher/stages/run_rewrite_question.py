from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import RewriteQuestion
from fedotllm.agents.utils import render
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_rewrite_question(state: ResearcherAgentState, inference: AIInference) -> ResearcherAgentState:
    state['question'] = RewriteQuestion.model_validate(
        inference.chat_completion(*render(get_settings().get("prompts.researcher.rewrite_question"),
                                          question=state['question']),
                                  structured=RewriteQuestion)).question
    return state
