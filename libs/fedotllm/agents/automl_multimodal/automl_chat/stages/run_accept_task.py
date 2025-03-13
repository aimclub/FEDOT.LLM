from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState

from fedotllm.log import get_logger
logger = get_logger()

def run_accept_task(state: AutoMLMultimodalAgentState):
    problem_description = ""
    for i in range(len(state['messages']) - 1, -1, -1):
        message = state['messages'][i]
        if message.name and message.name == "AutoMLMultimodalAgent":
            continue
        else:
            problem_description = message.content
            break
    if problem_description == "":
        raise ValueError("Problem description not found")
    
    state['description'] = problem_description
    return state

