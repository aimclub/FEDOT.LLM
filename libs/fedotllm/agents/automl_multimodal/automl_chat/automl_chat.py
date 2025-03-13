from functools import partial

from langgraph.graph import START, END, StateGraph

from fedotllm.agents.automl_multimodal.automl import AutoMLMultimodalAgent
from fedotllm.agents.automl_multimodal.automl_chat.stages.run_accept_task import run_accept_task
from fedotllm.agents.automl_multimodal.automl_chat.stages.run_send_message import run_send_message

from fedotllm.data import Dataset
from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState
from fedotllm.llm.inference import AIInference

from fedotllm.log import get_logger
logger = get_logger()

class AutoMLMultimodalAgentChat:
    def __init__(self, inference: AIInference, dataset: Dataset):
        self.inference = inference
        self.dataset = dataset
        self.automl = AutoMLMultimodalAgent(
            inference=self.inference, dataset=self.dataset)

    def create_graph(self):
        workflow = StateGraph(AutoMLMultimodalAgentState)
        workflow.add_node("accept_task", run_accept_task)
        workflow.add_node("AutoMLMultimodalAgent", self.automl.create_graph())
        workflow.add_node("send_message", partial(
            run_send_message, inference=self.inference))

        workflow.add_edge(START, "accept_task")
        workflow.add_edge("accept_task", "AutoMLMultimodalAgent")
        workflow.add_edge("AutoMLMultimodalAgent", "send_message")
        workflow.add_edge("send_message", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLMultimodalAgentChat"})
