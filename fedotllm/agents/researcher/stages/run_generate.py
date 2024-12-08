from agents.utils import render
from agents.researcher.structured import GenerateWithCitations
from settings.config_loader import get_settings
from agents.researcher.state import ResearcherAgentState
from llm.inference import AIInference


def run_generate(state: ResearcherAgentState, inference: AIInference):
    state["generation"] = inference.chat_completion(*render(get_settings().get("prompts.researcher.generate"),
                                                    question=state["question"], documents=state["documents"]),
                                                    structured=GenerateWithCitations)
    return state
