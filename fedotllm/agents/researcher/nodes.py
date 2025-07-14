from typing import List

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import (
    BoolAnswer,
    Citation,
    GenerateWithCitations,
    GradeAnswer,
    GradeDocuments,
    GradeHallucination,
    RewriteQuestion,
)
from fedotllm.agents.retrieve import RetrieveTool
from fedotllm.llm import AIInference, LiteLLMEmbeddings
from fedotllm.prompts.researcher import (
    generate_prompt,
    is_grounded_prompt,
    is_useful_prompt,
    retrieve_grader_prompt,
    rewrite_question_prompt,
)


def retrieve_documents(
    state: ResearcherAgentState, embeddings: LiteLLMEmbeddings
) -> ResearcherAgentState:
    retriever = RetrieveTool(embeddings=embeddings)
    if retriever.count() == 0:
        retriever.create_db_docs()
    retrieved = retriever.query_docs(state["question"])
    return Command(update={"retrieved": retrieved})


def grade_retrieve(
    state: ResearcherAgentState, inference: AIInference
) -> ResearcherAgentState:
    question = state["messages"][-1].content
    documents = state["retrieved"]["documents"]

    # Score each doc
    for i in range(len(documents)):
        document = state["retrieved"]["documents"][0][i]
        score = inference.create(
            retrieve_grader_prompt(question, document),
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


def generate_response(state: ResearcherAgentState, inference: AIInference):
    documents = "".join(
        [
            f'{metadatas["title"]}\n\nsource:"{metadatas["source"]}"\n\n{document}'
            for document, metadatas in zip(
                state["retrieved"]["documents"][0], state["retrieved"]["metadatas"][0]
            )
        ]
    )
    state["generation"] = inference.create(
        generate_prompt(documents, state["question"]),
        response_model=GenerateWithCitations,
    )
    return state


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
        inference.create(
            is_grounded_prompt(documents, state["generation"]),
            response_model=GradeHallucination,
        )
    )
    return grade.score == BoolAnswer.YES


def is_useful(state: ResearcherAgentState, inference: AIInference):
    grade = GradeAnswer.model_validate(
        inference.create(
            is_useful_prompt(state["generation"], state["question"]),
            response_model=GradeAnswer,
        )
    )
    return grade.score == BoolAnswer.YES


def render_answer(state: ResearcherAgentState):
    generation = state["generation"]
    answer = generation.answer
    citations: List[Citation] = generation.citations
    for citation in citations:
        answer = answer.replace(
            f"[{citation.number}]", f"[\\[{citation.number}\\]]({citation.url})"
        )

    state["answer"] = answer
    return Command(
        update={"messages": HumanMessage(content=answer, name="ResearcherAgent")}
    )


def is_continue(state: ResearcherAgentState):
    attempt = state.get("attempt", 0)
    if attempt:
        attempt += 1
        return True
    else:
        return False


def rewrite_question(
    state: ResearcherAgentState, inference: AIInference
) -> ResearcherAgentState:
    state["question"] = RewriteQuestion.model_validate(
        inference.create(
            rewrite_question_prompt(state["question"]),
            response_model=RewriteQuestion,
        )
    ).question
    return state
