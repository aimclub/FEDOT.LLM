from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from fedotllm.agents.automl.nodes import (
    evaluate,
    extract_metrics,
    fix_solution,
    generate_automl_config,
    generate_code,
    generate_report,
    if_bug,
    insert_templates,
    problem_reflection,
    run_tests,
    select_skeleton,
)
from fedotllm.data import Dataset
from fedotllm.llm import AIInference

from .state import AutoMLAgentState


class AutoMLAgent:
    def __init__(self, inference: AIInference, dataset: Dataset, workspace: Path):
        self.inference = inference
        self.dataset = dataset
        self.workspace = workspace

    def init_state(self, state: AutoMLAgentState):
        return Command(
            update={
                "reflection": None,
                "fedot_config": None,
                "skeleton": None,
                "raw_code": None,
                "code": None,
                "observation": None,
                "fix_attempts": 0,
                "metrics": "",
                "pipeline": "",
                "report": "",
            }
        )

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node("init_state", self.init_state)
        workflow.add_node(
            "problem_reflection",
            partial(problem_reflection, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node(
            "generate_automl_config",
            partial(
                generate_automl_config,
                inference=self.inference,
                dataset=self.dataset,
            ),
        )
        workflow.add_node(
            "select_skeleton",
            partial(select_skeleton, dataset=self.dataset, workspace=self.workspace),
        )
        workflow.add_node("insert_templates", insert_templates)
        workflow.add_node(
            "generate_code",
            partial(generate_code, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node("evaluate_main", partial(evaluate, workspace=self.workspace))
        workflow.add_node(
            "fix_solution_main",
            partial(fix_solution, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node(
            "run_tests",
            partial(run_tests, workspace=self.workspace, inference=self.inference),
        )
        workflow.add_node(
            "extract_metrics", partial(extract_metrics, workspace=self.workspace)
        )
        workflow.add_node(
            "generate_report", partial(generate_report, inference=self.inference)
        )

        workflow.add_edge(START, "init_state")
        workflow.add_edge("init_state", "problem_reflection")
        workflow.add_edge("problem_reflection", "generate_automl_config")
        workflow.add_edge("generate_automl_config", "select_skeleton")
        workflow.add_edge("select_skeleton", "generate_code")
        workflow.add_edge("generate_code", "insert_templates")
        workflow.add_conditional_edges(
            "insert_templates",
            lambda state: state["code"] is None,
            {True: "generate_code", False: "evaluate_main"},
        )
        workflow.add_conditional_edges(
            "evaluate_main",
            if_bug,
            {True: "fix_solution_main", False: "run_tests"},
        )
        workflow.add_edge("fix_solution_main", "insert_templates")
        workflow.add_conditional_edges(
            "run_tests",
            if_bug,
            {True: "fix_solution_main", False: "extract_metrics"},
        )
        workflow.add_edge("extract_metrics", "generate_report")
        workflow.add_edge("generate_report", END)
        return workflow.compile().with_config(config={"run_name": "AutoMLAgent"})
