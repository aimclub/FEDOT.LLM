from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from fedot_llm.ai.chains.ready_chains import PredictChain
from fedot_llm.data import Dataset


class AutoMLAgent:
    def __init__(self, llm: BaseChatModel, dataset: Dataset):
        self.llm = llm
        self.dataset = dataset

    @property
    def as_tool(self):
        @tool()
        def automl(dataset_description: Annotated[str, "Description of the dataset and task to be performed"]):
            """Uses AutoML to create a model that performs classification, regression, or time series tasks."""
            return PredictChain(model=self.llm, dataset=self.dataset).invoke(
                {"dataset_description": dataset_description})

        return automl

    def create_graph(self):
        return PredictChain(model=self.llm, dataset=self.dataset)
