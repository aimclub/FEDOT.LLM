import fedotllm.prompting.prompts as prompts
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import BoolAnswer, GradeAnswer
from fedotllm.llm import LiteLLMModel


def is_useful(state: ResearcherAgentState, llm: LiteLLMModel):
    grade = llm.create(
        response_model=GradeAnswer,
        messages=prompts.researcher.is_useful_prompt(
            state["generation"], state["question"]
        ),
    )
    return grade.score == BoolAnswer.YES
