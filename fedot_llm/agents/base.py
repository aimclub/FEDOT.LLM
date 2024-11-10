from langgraph.graph.state import CompiledStateGraph


class Agent:
    def create_graph(self) -> CompiledStateGraph:
        raise NotImplementedError