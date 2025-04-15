import fedotllm.prompting.prompts as prompts
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import BoolAnswer, GradeDocuments
from fedotllm.llm import LiteLLMModel


def run_retrieve_grader(
    state: ResearcherAgentState, llm: LiteLLMModel
) -> ResearcherAgentState:
    question = state["question"]
    documents = state["retrieved"]["documents"]

    # Score each doc
    for i in range(len(documents)):
        document = state["retrieved"]["documents"][0][i]
        score = llm.create(
            messages=prompts.researcher.retrieve_grader_prompt(question, document),
            response_model=GradeDocuments,
        )

        grade = score.score
        if grade == BoolAnswer.NO:
            del state["retrieved"]["ids"][0][i]
            del state["retrieved"]["distances"][0][i]
            del state["retrieved"]["documents"][0][i]
            del state["retrieved"]["metadatas"][0][i]
        else:
            continue
    return state
