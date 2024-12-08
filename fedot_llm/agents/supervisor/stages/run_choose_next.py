from fedot_llm.agents.supervisor.state import SupervisorState
from fedot_llm.llm.inference import AIInference
from settings.config_loader import get_settings
from fedot_llm.agents.supervisor.structured import ChooseNext
from fedot_llm.agents.utils import render


def run_choose_next(state: SupervisorState, inference: AIInference):
    messages = state["messages"]
    if isinstance(messages, list):
        messages_str = "\n".join([f"{m.name}: {m.content}" for m in messages])
    else:
        messages_str = f"{messages.name}: {messages.content}"

    response = inference.chat_completion(*render(get_settings().prompts.supervisor.choose_next,
                                                  {"messages": messages_str}), structured=ChooseNext)
    state["next"] = response.next
    return state
