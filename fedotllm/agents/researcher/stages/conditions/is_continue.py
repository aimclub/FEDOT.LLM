from agents.researcher.state import ResearcherAgentState


def is_continue(state: ResearcherAgentState):
    if state["attempt"] < 3:
        state["attempt"] += 1
        return True
    else:
        return False
