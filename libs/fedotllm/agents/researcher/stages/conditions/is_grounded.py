from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GradeHallucination, BoolAnswer
from fedotllm.agents.utils import render
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def is_grounded(state: ResearcherAgentState, inference: AIInference):
    documents = ''.join([
        f"{metadatas['title']}\n\nsource:\"{metadatas['source']}\"\n\n{document}"
        for document, metadatas in zip(state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0])
    ])
    grade = GradeHallucination.model_validate(
        inference.chat_completion(*render(get_settings().get("prompts.researcher.is_grounded"),
                                          generation=state["generation"], documents=documents),
                                  structured=GradeHallucination))
    return grade.score == BoolAnswer.YES
