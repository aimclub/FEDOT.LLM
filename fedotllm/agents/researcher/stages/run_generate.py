import fedotllm.prompting.prompts as prompts
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GenerateWithCitations
from fedotllm.llm import LiteLLMModel


def run_generate(state: ResearcherAgentState, llm: LiteLLMModel):
    documents = "".join(
        [
            f'{metadatas["title"]}\n\nsource:"{metadatas["source"]}"\n\n{document}'
            for document, metadatas in zip(
                state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0]
            )
        ]
    )
    state["generation"] = llm.create(
        messages=prompts.researcher.generate_prompt(documents, state["question"]),
        response_model=GenerateWithCitations,
    )
    return state
