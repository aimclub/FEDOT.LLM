from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GenerateWithCitations
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


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
        prompts.researcher.generate_prompt(documents, state["question"]),
        structured=GenerateWithCitations,
    )
    return state
