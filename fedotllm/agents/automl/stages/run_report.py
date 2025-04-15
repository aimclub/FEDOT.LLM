import fedotllm.prompting.prompts as prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm import LiteLLMModel


def run_report(state: AutoMLAgentState, llm: LiteLLMModel):
    if state["solutions"][-1]["code"] and state["pipeline"]:
        response = llm.query(
            prompts.automl.reporter_prompt(
                description=state["description"],
                metrics=state["metrics"],
                pipeline=state["pipeline"],
                code=state["solutions"][-1]["code"],
            )
        )
    else:
        response = "Solution not found. Please try again."
    state["report"] = response
    return state
