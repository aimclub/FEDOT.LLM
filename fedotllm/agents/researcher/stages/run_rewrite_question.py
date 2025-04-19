from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import RewriteQuestion
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


def run_rewrite_question(
    state: ResearcherAgentState, inference: AIInference
) -> ResearcherAgentState:
    state["question"] = RewriteQuestion.model_validate(
        inference.chat_completion(
            prompts.researcher.rewrite_question_prompt(state["question"]),
            structured=RewriteQuestion,
        )
    ).question
    return state
