import fedotllm.prompting.prompts as prompts
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import RewriteQuestion
from fedotllm.llm import LiteLLMModel


def run_rewrite_question(
    state: ResearcherAgentState, llm: LiteLLMModel
) -> ResearcherAgentState:
    rewrite_question = llm.create(
        messages=prompts.researcher.rewrite_question_prompt(state["question"]),
        response_model=RewriteQuestion,
    )
    state["question"] = rewrite_question.question
    return state
