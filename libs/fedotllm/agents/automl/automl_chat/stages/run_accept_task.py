from fedotllm.agents.automl.state import AutoMLAgentState


def run_accept_task(state: AutoMLAgentState):
    problem_description = ""
    for i in range(len(state['messages']) - 1, -1, -1):
        message = state['messages'][i]
        if message.name and message.name == "AutoMLAgent":
            continue
        else:
            problem_description = message.content
            break
    if problem_description == "":
        raise ValueError("Problem description not found")
    state['description'] = problem_description
    return state
