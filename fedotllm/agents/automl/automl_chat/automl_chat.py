from pathlib import Path

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.automl.automl_chat.stages.run_accept_task import run_accept_task
from fedotllm.agents.automl.automl_chat.stages.run_send_message import run_send_message
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.data import Dataset
from fedotllm.llm import AIInference


class AutoMLAgentChat:
    def __init__(self, inference: AIInference, dataset: Dataset, workspace: Path):
        self.inference = inference
        self.dataset = dataset
        self.workspace = workspace
        self.automl = AutoMLAgent(inference=self.inference, dataset=self.dataset, workspace=self.workspace)

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node("accept_task", run_accept_task)
        workflow.add_node("AutoMLAgent", self.automl.create_graph())
        workflow.add_node("send_message", run_send_message)

        workflow.add_edge(START, "accept_task")
        workflow.add_edge("accept_task", "AutoMLAgent")
        workflow.add_edge("AutoMLAgent", "send_message")
        workflow.add_edge("send_message", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLAgentChat"})
