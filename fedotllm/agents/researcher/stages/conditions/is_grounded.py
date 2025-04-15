import fedotllm.prompting.prompts as prompts
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import BoolAnswer, GradeHallucination
from fedotllm.llm import LiteLLMModel


def is_grounded(state: ResearcherAgentState, llm: LiteLLMModel):
    documents = "".join(
        [
            f'{metadatas["title"]}\n\nsource:"{metadatas["source"]}"\n\n{document}'
            for document, metadatas in zip(
                state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0]
            )
        ]
    )
    grade = llm.create(
        messages=prompts.researcher.is_grounded_prompt(documents, state["generation"]),
        response_model=GradeHallucination,
    )
    return grade.score == BoolAnswer.YES
