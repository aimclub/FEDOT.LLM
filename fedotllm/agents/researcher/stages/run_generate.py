from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GenerateWithCitations
from fedotllm.agents.utils import render
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_generate(state: ResearcherAgentState, inference: AIInference):
    documents = "".join(
        [
            f'{metadatas["title"]}\n\nsource:"{metadatas["source"]}"\n\n{document}'
            for document, metadatas in zip(
                state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0]
            )
        ]
    )
    state["generation"] = inference.chat_completion(
        *render(
            get_settings().get("prompts.researcher.generate"),
            question=state["question"],
            documents=documents,
        ),
        structured=GenerateWithCitations,
    )
    return state
