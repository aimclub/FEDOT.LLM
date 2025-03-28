from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GradeHallucination, BoolAnswer
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


def is_grounded(state: ResearcherAgentState, inference: AIInference):
    documents = "".join(
        [
            f'{metadatas["title"]}\n\nsource:"{metadatas["source"]}"\n\n{document}'
            for document, metadatas in zip(
                state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0]
            )
        ]
    )
    grade = GradeHallucination.model_validate(
        inference.chat_completion(
            prompts.researcher.is_grounded_prompt(documents, state["generation"]),
            structured=GradeHallucination,
        )
    )
    return grade.score == BoolAnswer.YES
