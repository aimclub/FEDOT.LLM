import re
from enum import Enum
from functools import partial

from langchain_core.messages import convert_to_openai_messages
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from fedotllm.agents.base import Agent, FedotLLMAgentState
from fedotllm.configs.schema import AppConfig
from fedotllm.llm import AIInference
from fedotllm.prompts.supervisor import choose_next_prompt


class NextAgent(str, Enum):
    FINISH = "finish"
    RESEARCHER = "researcher"
    AUTOML = "automl"


class SupervisorState(FedotLLMAgentState):
    next: NextAgent


class SupervisorAgent(Agent):
    def __init__(
        self,
        config: AppConfig,
        automl_agent: Runnable,
        researcher_agent: Runnable,
    ):
        self.inference = AIInference(config.llm, config.session_id)
        self.researcher_agent = researcher_agent
        self.automl_agent = automl_agent

    def create_graph(self):
        workflow = StateGraph(SupervisorState)

        workflow.add_node("choose_next", partial(router_node, inference=self.inference))
        workflow.add_node("researcher", self.researcher_agent)
        workflow.add_node("automl", self.automl_agent)

        def finish_execution(state: SupervisorState):
            return state

        workflow.add_node("finish", finish_execution)

        workflow.add_edge(START, "choose_next")
        workflow.add_edge("researcher", "choose_next")
        workflow.add_edge("automl", "finish")
        workflow.add_edge("finish", END)
        return workflow.compile().with_config(run_name=SupervisorAgent)


class ChooseNext(BaseModel):
    next: NextAgent = Field(
        ...,
        description="""The next agent to act or finish.
finish - the conversation is finished.
automl - responsible for automl tasks, can building machine learning models, ML pipelines, **build**
researcher - responsible for QA about the Fedot framework.""",
    )


def router_node(
    state: SupervisorState,
    inference: AIInference,
) -> Command:
    """
    Router node to choose the next agent based on the current state and inference.
    """

    messages = convert_to_openai_messages(state["messages"])
    messages = [messages] if isinstance(messages, dict) else messages
    messages.append({"role": "user", "content": choose_next_prompt()})

    response = inference.query(messages)
    response = response.strip().lower()
    match = re.search(r"\b(automl|researcher|finish)\b", response)
    if not match:
        raise ValueError(
            f"Invalid response from inference: '{response}'. Expected 'automl', 'researcher', or 'finish'."
        )
    return Command(goto=match.group(0))
