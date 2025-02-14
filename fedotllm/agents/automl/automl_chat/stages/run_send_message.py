from pathlib import Path

from langchain_core.messages import HumanMessage

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_send_message(state: AutoMLAgentState, inference: AIInference):
    if state['solutions'][-1]['code'] is None:
        state['messages'] = [HumanMessage(
            content="Solution not found. Please try again.", name="AutoMLAgent")]
    else:
        message = inference.chat_completion(get_settings().prompts.automl.automl_chat.run_send_message.user.format(
            description=state['description'],
            metrics=state['metrics'],
            pipeline=state['pipeline'],
            code=state['solutions'][-1]['code']
        ))
        state['messages'] = [HumanMessage(
            content=message.content, name="AutoMLAgent")]
    return state
