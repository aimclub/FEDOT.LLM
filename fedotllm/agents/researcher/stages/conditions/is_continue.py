from fedotllm.agents.researcher.state import ResearcherAgentState


def is_continue(state: ResearcherAgentState):
    attempt = state.get("attempt", 0)
    if attempt:
        attempt += 1
        return True
    else:
        return False
