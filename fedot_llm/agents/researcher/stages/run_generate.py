from fedot_llm.agents.researcher.state import ResearcherAgentState
from fedot_llm.llm.inference import AIInference
from fedot_llm.agents.researcher.structured import GenerateWithCitations
from fedot_llm.agents.utils import render
from settings.config_loader import get_settings


def run_generate(state: ResearcherAgentState, inference: AIInference):
    state["generation"] = inference.chat_completion(*render(get_settings().get("prompts.researcher.generate"),
                                                    question=state["question"], documents=state["documents"]),
                                           structured=GenerateWithCitations)
    return state
