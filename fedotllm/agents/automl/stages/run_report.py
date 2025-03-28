from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


def run_report(state: AutoMLAgentState, inference: AIInference):
    if state["solutions"][-1]["code"] and state["pipeline"]:
        response = inference.chat_completion(
            prompts.automl.reporter_prompt(
                description=state["description"],
                metrics=state["metrics"],
                pipeline=state["pipeline"],
                code=state["solutions"][-1]["code"],
            )
        ).content
    else:
        response = "Solution not found. Please try again."
    state["report"] = response
    return state
