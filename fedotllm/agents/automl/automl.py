from functools import partial

from langgraph.graph import END, START, StateGraph

from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference

from .stages.conditions.if_bug import if_bug
from .stages.run_codegen import run_codegen
from .stages.run_evaluate import run_evaluate
from .stages.run_extract_metrics import run_extract_metrics
from .stages.run_fix_solution import run_fix_solution
from .stages.run_generate_automl_config import run_generate_automl_config
from .stages.run_insert_templates import run_insert_templates
from .stages.run_problem_reflection import run_problem_reflection
from .stages.run_select_skeleton import run_select_skeleton
from .stages.run_report import run_report
from .state import AutoMLAgentState


class AutoMLAgent:
    def __init__(self, inference: AIInference, dataset: Dataset):
        self.inference = inference
        self.dataset = dataset

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node(
            "problem_reflection",
            partial(
                run_problem_reflection, inference=self.inference, dataset=self.dataset
            ),
        )
        workflow.add_node(
            "generate_automl_config",
            partial(
                run_generate_automl_config,
                inference=self.inference,
                dataset=self.dataset,
            ),
        )
        workflow.add_node(
            "select_skeleton", partial(run_select_skeleton, dataset=self.dataset)
        )
        workflow.add_node("insert_templates", run_insert_templates)
        workflow.add_node(
            "codegen",
            partial(run_codegen, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node("evaluate_main", run_evaluate)
        workflow.add_node(
            "fix_solution_main",
            partial(run_fix_solution, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node("extract_metrics", run_extract_metrics)
        workflow.add_node("report_node", partial(run_report, inference=self.inference))

        workflow.add_edge(START, "problem_reflection")
        workflow.add_edge("problem_reflection", "generate_automl_config")
        workflow.add_edge("generate_automl_config", "select_skeleton")
        workflow.add_edge("select_skeleton", "codegen")
        workflow.add_edge("codegen", "insert_templates")
        workflow.add_conditional_edges(
            "insert_templates",
            lambda state: state["solutions"][-1]["code"] is None,
            {True: "codegen", False: "evaluate_main"},
        )
        workflow.add_conditional_edges(
            "evaluate_main",
            if_bug,
            {True: "fix_solution_main", False: "extract_metrics"},
        )
        workflow.add_edge("fix_solution_main", "insert_templates")
        workflow.add_edge("extract_metrics", "report_node")
        workflow.add_edge("report_node", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLAgent"})
