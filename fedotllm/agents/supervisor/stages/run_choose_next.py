import fedotllm.prompting.prompts as prompts
from fedotllm.agents.supervisor.state import SupervisorState
from fedotllm.agents.supervisor.structured import ChooseNext
from fedotllm.llm import LiteLLMModel


def run_choose_next(state: SupervisorState, llm: LiteLLMModel):
    messages = state["messages"]
    if isinstance(messages, list):
        messages_str = "\n".join([f"{m.name}: {m.content}" for m in messages])
    else:
        messages_str = f"{messages.name}: {messages.content}"

    response = llm.create(
        messages=[
            {
                "role": "user",
                "content": prompts.supervisor.choose_next_prompt(messages_str),
            }
        ],
        response_model=ChooseNext,
    )

    state["next"] = response.next
    return state
