from enum import Enum
from functools import partial

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from fedotllm.agents.base import Agent, FedotLLMAgentState
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
        inference: AIInference,
        automl_agent: Runnable,
        researcher_agent: Runnable,
    ):
        self.inference = inference
        self.researcher_agent = researcher_agent
        self.automl_agent = automl_agent

    def create_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("choose_next", partial(choose_next, inference=self.inference))
        workflow.add_node("researcher", self.researcher_agent)
        workflow.add_node("automl", self.automl_agent)

        def finish_execution(state: SupervisorState):
            return state

        workflow.add_node("finish", finish_execution)

        workflow.add_edge(START, "choose_next")
        workflow.add_edge("researcher", "choose_next")
        workflow.add_edge("automl", "finish")
        workflow.add_edge("finish", END)
        return workflow.compile().with_config(config={"run_name": "SupervisorAgent"})


class ChooseNext(BaseModel):
    next: NextAgent = Field(
        ...,
        description="""The next agent to act or finish.
finish - the conversation is finished.
automl - responsible for automl tasks, can building machine learning models, ML pipelines, **build**
researcher - responsible for QA about the Fedot framework.""",
    )


def choose_next(state: SupervisorState, inference: AIInference):
    messages = state["messages"]
    if isinstance(messages, list):
        messages_str = "\n".join([f"{m.name}: {m.content}" for m in messages])
    else:
        messages_str = f"{messages.name}: {messages.content}"

    response = inference.create(
        choose_next_prompt(messages_str),
        response_model=ChooseNext,
    )
    return Command(goto=response.next)
