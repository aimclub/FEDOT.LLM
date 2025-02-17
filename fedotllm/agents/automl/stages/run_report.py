from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_report(state: AutoMLAgentState, inference: AIInference):
    if state['solutions'][-1]['code'] and state['pipeline']:
        response = inference.chat_completion(get_settings().prompts.automl.automl_chat.run_send_message.user.format(
            description=state['description'],
            metrics=state['metrics'],
            pipeline=state['pipeline'],
            code=state['solutions'][-1]['code']
        )).content
    else:
        response = "Solution not found. Please try again."
    state['report'] = response
    return state
    
