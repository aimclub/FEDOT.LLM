from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState

from fedot_llm.ai.agents.base import extract_calls
from fedot_llm.ai.chains.ready_chains import PredictChain
from fedot_llm.data import Dataset


class AutoMLAgent:
    def __init__(self, llm: BaseChatModel, dataset: Dataset):
        self.llm = llm
        self.dataset = dataset
        self.as_graph = self.create_graph()

    def run(self, dataset_description: str):
        return self.as_graph.invoke({"dataset_description": dataset_description})

    @property
    def as_tool(self):
        @tool("AutoMLAgent")
        def automl(dataset_description: Annotated[str, "Description of the dataset and task to be performed"]):
            """Uses AutoML to create a model that performs classification, regression, or time series tasks."""
            return self.run(dataset_description)

        return automl

    def invoke_chain(self, state: MessagesState):
        if calls := extract_calls(state):
            if (args := calls.get("AutoMLAgent", None).get("args", None)) is None:
                raise ValueError("AutoMLAgent call not found or has no args")
            response = PredictChain(model=self.llm, dataset=self.dataset).invoke(args)
            response_msg = HumanMessage(
                content=str(response),
                name="AutoMLAgent", )
            return {'messages': [response_msg], 'next': 'FINISH', 'args': {}}
        else:
            raise ValueError("Not found AutoMLAgent call.")

    def create_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("invoke_chain", self.invoke_chain)

        workflow.add_edge(START, "invoke_chain")
        workflow.add_edge("invoke_chain", END)

        return workflow.compile()
