from langchain.schema import Document

from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import GradeDocuments, BoolAnswer
from fedotllm.agents.utils import render
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_retrieve_grader(state: ResearcherAgentState, inference: AIInference) -> ResearcherAgentState:
    question = state["question"]
    documents = state["retrieved"]['documents']

    # Score each doc
    for i in range(len(documents)):
        document = state['retrieved']['documents'][0][i]
        score = GradeDocuments.model_validate(
            inference.chat_completion(*render(get_settings().get("prompts.researcher.retrieve_grader"),
                                              question=question, document=document),
                                      structured=GradeDocuments))

        grade = score.score
        if grade == BoolAnswer.NO:
            del state['retrieved']['ids'][0][i]
            del state['retrieved']['distances'][0][i]
            del state['retrieved']['documents'][0][i]
            del state['retrieved']['metadatas'][0][i]
        else:
            continue
    return state
