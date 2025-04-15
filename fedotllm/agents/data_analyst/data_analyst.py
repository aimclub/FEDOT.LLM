from pathlib import Path

from langgraph.graph import START, StateGraph
from omegaconf import DictConfig

from fedotllm.agents.base import Agent
from fedotllm.agents.data_analyst.nodes import act, plan, think
from fedotllm.agents.data_analyst.schema import Memory
from fedotllm.agents.data_analyst.state import DataAnalystAgentState
from fedotllm.environments.jupyter import JupyterExecutor
from fedotllm.lib import LIBREG
from fedotllm.llm import LiteLLMModel
from fedotllm.tools.finish import FinishTool
from fedotllm.tools.jupyter import JupyterTool
from fedotllm.tools.planning import PlanningTool
from fedotllm.tools.tools_collection import ToolCollection
from fedotllm.utils import unpack_omega_config


class DataAnalystAgent(Agent):
    def __init__(self, config: DictConfig, session_id: str, workspace: Path):
        self.llm = LiteLLMModel(
            **unpack_omega_config(config.llm), session_id=session_id
        )
        self.memory = Memory(llm=self.llm)
        self.available_tools = ToolCollection(
            PlanningTool(),
            JupyterTool(executor=JupyterExecutor(workspace)),
            FinishTool(),
        )
        self.library_functions = LIBREG.to_markdown()

    async def plan(self, state: DataAnalystAgentState):
        return await plan(self, state)

    async def act(self, state: DataAnalystAgentState):
        return await act(self, state)

    async def think(self, state: DataAnalystAgentState):
        return await think(self, state)

    def initialize_state(
        self, problem_description: str, workspace: Path, task_path: Path
    ) -> dict:
        """Initialize the state with required attributes"""
        return {
            "memory": self.memory,
            "tool_calls": [],
            "available_tools": self.available_tools,
            "problem_description": problem_description,
            "library_functions": self.library_functions,
            "workspace": workspace,
            "task_path": task_path,
            "messages": [],
        }

    def create_graph(self):
        workflow = StateGraph(DataAnalystAgentState)
        workflow.add_node("plan", self.plan)
        workflow.add_node("act", self.act)
        workflow.add_node("think", self.think)

        workflow.add_edge(START, "plan")
        compiled_workflow = workflow.compile()

        return compiled_workflow
