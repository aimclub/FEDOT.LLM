from functools import partial

from langgraph.graph import StateGraph, START, END

from fedotllm.data import Dataset
from fedotllm.agents.automl_multimodal.stages.run_train_automl import run_train_automl
from fedotllm.agents.automl_multimodal.stages.run_generate_automl_config import run_generate_automl_config
from fedotllm.agents.automl_multimodal.stages.run_problem_reflection import run_problem_reflection
from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState
from fedotllm.llm.inference import AIInference

from fedotllm.log import get_logger
logger = get_logger()

class AutoMLMultimodalAgent:
    def __init__(self, inference: AIInference, dataset: Dataset):
        self.inference = inference
        self.dataset = dataset

    def create_graph(self):
        workflow = StateGraph(AutoMLMultimodalAgentState)
        workflow.add_node("problem_reflection", partial(
            run_problem_reflection, inference=self.inference, dataset=self.dataset))
        workflow.add_node("generate_automl_config", partial(
            run_generate_automl_config, inference=self.inference, dataset=self.dataset))
        workflow.add_node("evaluate_main", partial(
            run_train_automl, dataset=self.dataset))

        workflow.add_edge(START, "problem_reflection")
        workflow.add_edge("problem_reflection", "generate_automl_config")
        workflow.add_edge("generate_automl_config", "evaluate_main")
        workflow.add_edge("evaluate_main", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLMultimodalAgent"})
