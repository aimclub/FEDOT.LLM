from fedotllm.agents.supervisor.state import SupervisorState
from fedotllm.agents.supervisor.structured import ChooseNext
from fedotllm.llm.inference import AIInference
import fedotllm.prompts as prompts


def run_choose_next(state: SupervisorState, inference: AIInference):
    messages = state["messages"]
    if isinstance(messages, list):
        messages_str = "\n".join([f"{m.name}: {m.content}" for m in messages])
    else:
        messages_str = f"{messages.name}: {messages.content}"

    response = inference.chat_completion(
        prompts.supervisor.choose_next_prompt(messages_str),
        structured=ChooseNext,
    )
    state["next"] = response.next
    return state
