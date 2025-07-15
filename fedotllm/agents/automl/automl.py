from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.automl.nodes import (
    evaluate,
    extract_metrics,
    fix_solution,
    generate_automl_config,
    generate_code,
    generate_report,
    if_bug,
    init_state,
    insert_templates,
    problem_reflection,
    run_tests,
    select_skeleton,
)
from fedotllm.configs.schema import AppConfig
from fedotllm.data import Dataset
from fedotllm.llm import AIInference

from .state import AutoMLAgentState

INIT_STATE = "init_state"
PROBLEM_REFLECTION = "problem_reflection"
GENERATE_AUTOML_CONFIG = "generate_automl_config"
SELECT_SKELETON = "select_skeleton"
INSERT_TEMPLATE_FIRST = "insert_templates_first"
GENERATE_CODE = "generate_code"
EVALUATE_FIRST = "evaluate_first"
FIX_SOLUTION = "fix_solution"
RUN_TESTS = "run_tests"
INSERT_TEMPLATE_FINAL = "insert_templates_final"
EVALUATE_FINAL = "evaluate_final"
EXTRACT_METRICS = "extract_metrics"
GENERATE_REPORT = "generate_report"


class AutoMLAgent:
    def __init__(
        self, config: AppConfig, dataset_path: str | Path, workspace: str | Path
    ):
        self.config = config
        self.inference = AIInference(config.llm, config.session_id)
        self.dataset = Dataset.from_path(dataset_path)
        self.workspace = Path(workspace)

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node(INIT_STATE, init_state)
        workflow.add_node(
            PROBLEM_REFLECTION,
            partial(problem_reflection, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node(
            GENERATE_AUTOML_CONFIG,
            partial(
                generate_automl_config,
                inference=self.inference,
                dataset=self.dataset,
            ),
        )
        workflow.add_node(
            SELECT_SKELETON,
            partial(
                select_skeleton,
                app_config=self.config,
                dataset=self.dataset,
                workspace=self.workspace,
            ),
        )
        workflow.add_node(
            INSERT_TEMPLATE_FIRST,
            partial(insert_templates, app_config=self.config, run_mode="test"),
        )
        workflow.add_node(
            GENERATE_CODE,
            partial(generate_code, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node(EVALUATE_FIRST, partial(evaluate, workspace=self.workspace))
        workflow.add_node(
            FIX_SOLUTION,
            partial(fix_solution, inference=self.inference, dataset=self.dataset),
        )
        workflow.add_node(
            RUN_TESTS,
            partial(run_tests, workspace=self.workspace, inference=self.inference),
        )
        workflow.add_node(
            INSERT_TEMPLATE_FINAL,
            partial(insert_templates, app_config=self.config, run_mode="final"),
        )
        workflow.add_node(EVALUATE_FINAL, partial(evaluate, workspace=self.workspace))
        workflow.add_node(
            EXTRACT_METRICS, partial(extract_metrics, workspace=self.workspace)
        )
        workflow.add_node(
            GENERATE_REPORT, partial(generate_report, inference=self.inference)
        )

        workflow.add_edge(START, INIT_STATE)
        workflow.add_edge(INIT_STATE, PROBLEM_REFLECTION)
        workflow.add_edge(PROBLEM_REFLECTION, GENERATE_AUTOML_CONFIG)
        workflow.add_edge(GENERATE_AUTOML_CONFIG, SELECT_SKELETON)
        workflow.add_edge(SELECT_SKELETON, GENERATE_CODE)
        workflow.add_edge(GENERATE_CODE, INSERT_TEMPLATE_FIRST)
        workflow.add_conditional_edges(
            INSERT_TEMPLATE_FIRST,
            lambda state: state["code"] is None,
            {True: GENERATE_CODE, False: EVALUATE_FIRST},
        )
        workflow.add_conditional_edges(
            EVALUATE_FIRST,
            partial(if_bug, app_config=self.config),
            {True: FIX_SOLUTION, False: RUN_TESTS},
        )
        workflow.add_edge(FIX_SOLUTION, INSERT_TEMPLATE_FIRST)
        workflow.add_conditional_edges(
            RUN_TESTS,
            partial(if_bug, app_config=self.config),
            {True: FIX_SOLUTION, False: INSERT_TEMPLATE_FINAL},
        )
        workflow.add_edge(INSERT_TEMPLATE_FINAL, EVALUATE_FINAL)
        workflow.add_edge(EVALUATE_FINAL, EXTRACT_METRICS)
        workflow.add_edge(EXTRACT_METRICS, GENERATE_REPORT)
        workflow.add_edge(GENERATE_REPORT, END)
        return workflow.compile().with_config(run_name="AutoMLAgent")
