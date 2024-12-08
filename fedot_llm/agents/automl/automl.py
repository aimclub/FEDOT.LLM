from functools import partial

from langgraph.graph import END, START, StateGraph

from fedot_llm.agents.automl.stages.conditions.if_bug import if_bug
from fedot_llm.agents.automl.stages.run_codegen import run_codegen
from fedot_llm.agents.automl.stages.run_evaluate import run_evaluate
from fedot_llm.agents.automl.stages.run_fix_solution import run_fix_solution
from fedot_llm.agents.automl.stages.run_generate_fedot_config import \
    run_generate_fedot_config
from fedot_llm.agents.automl.stages.run_insert_templates import \
    run_insert_templates
from fedot_llm.agents.automl.stages.run_problem_reflection import \
    run_problem_reflection
from fedot_llm.agents.automl.stages.run_save_results import run_save_results
from fedot_llm.agents.automl.stages.run_select_skeleton import \
    run_select_skeleton
from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.agents.base import Agent
from fedot_llm.llm.inference import AIInference


class AutoMLAgent(Agent):
    def __init__(self, inference: AIInference):
        self.inference = inference

    def create_graph(self):
        workflow = StateGraph(AutoMLAgentState)
        workflow.add_node("problem_reflection", partial(
            run_problem_reflection, inference=self.inference))
        workflow.add_node("generate_fedot_config", partial(
            run_generate_fedot_config, inference=self.inference))
        workflow.add_node("select_skeleton", run_select_skeleton)
        workflow.add_node("insert_templates", run_insert_templates)
        workflow.add_node("codegen", partial(
            run_codegen, inference=self.inference))
        workflow.add_node("evaluate_main", partial(run_evaluate, stage='main'))
        workflow.add_node("fix_solution_main", partial(
            run_fix_solution, inference=self.inference))
        workflow.add_node("fix_solution_deploy", partial(
            run_fix_solution, inference=self.inference))
        workflow.add_node("evaluate_deploy", partial(
            run_evaluate, stage='deploy'))
        workflow.add_node("save_results", run_save_results)
        
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
                False: "evaluate_deploy"
            }
        )
        workflow.add_edge("fix_solution_main", "evaluate_main")
        workflow.add_conditional_edges(
            "evaluate_deploy",
            if_bug,
            {
                True: "fix_solution_deploy",
                False: "save_results"
            }
        )
        workflow.add_edge("fix_solution_deploy", "evaluate_deploy")
        workflow.add_edge("save_results", END)
        return workflow.compile()
