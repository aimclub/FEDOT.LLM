from functools import partial

from langgraph.graph import StateGraph, START, END

from fedotllm.agents.automl.data.data import Dataset
from fedotllm.agents.automl.stages.conditions.if_bug import if_bug
from fedotllm.agents.automl.stages.run_codegen import run_codegen
from fedotllm.agents.automl.stages.run_evaluate import run_evaluate
from fedotllm.agents.automl.stages.run_extract_metrics import run_extract_metrics
from fedotllm.agents.automl.stages.run_fix_solution import run_fix_solution
from fedotllm.agents.automl.stages.run_generate_fedot_config import run_generate_fedot_config
from fedotllm.agents.automl.stages.run_insert_templates import run_insert_templates
from fedotllm.agents.automl.stages.run_problem_reflection import run_problem_reflection
from fedotllm.agents.automl.stages.run_save_results import run_save_results
from fedotllm.agents.automl.stages.run_select_skeleton import run_select_skeleton
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm.inference import AIInference


class AutoMLAgent:
    def __init__(self, inference: AIInference, dataset: Dataset):
        self.inference = inference
        self.dataset = dataset

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node("problem_reflection", partial(
            run_problem_reflection, inference=self.inference, dataset=self.dataset))
        workflow.add_node("generate_fedot_config", partial(
            run_generate_fedot_config, inference=self.inference, dataset=self.dataset))
        workflow.add_node("select_skeleton", run_select_skeleton)
        workflow.add_node("insert_templates", run_insert_templates)
        workflow.add_node("codegen", partial(
            run_codegen, inference=self.inference, dataset=self.dataset))
        workflow.add_node("evaluate_main", run_evaluate)
        workflow.add_node("fix_solution_main", partial(
            run_fix_solution, inference=self.inference, dataset=self.dataset))
        workflow.add_node("save_results", run_save_results)
        workflow.add_node("extract_metrics", run_extract_metrics)

        workflow.add_edge(START, "problem_reflection")
        workflow.add_edge("problem_reflection", "generate_fedot_config")
        workflow.add_edge("generate_fedot_config", "select_skeleton")
        workflow.add_edge("select_skeleton", "codegen")
        workflow.add_edge("codegen", "insert_templates")
        workflow.add_conditional_edges(
            "insert_templates",
            lambda state: state['solutions'][-1]['code'] is None,
            {
                True: "codegen",
                False: "evaluate_main"
            }
        )
        workflow.add_conditional_edges(
            "evaluate_main",
            if_bug,
            {
                True: "fix_solution_main",
                False: "save_results"
            }
        )
        workflow.add_edge("fix_solution_main", "insert_templates")
        workflow.add_edge("save_results", "extract_metrics")
        workflow.add_edge("extract_metrics", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLAgent"})
