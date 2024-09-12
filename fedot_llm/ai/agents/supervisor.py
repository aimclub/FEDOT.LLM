from langchain.chat_models.base import BaseChatModel
from langgraph.prebuilt import create_react_agent
from transformers import pipeline

from fedot_llm.ai.agents.automl import AutoMLAgent
from fedot_llm.ai.agents.researcher.researcher import ResearcherAgent
from fedot_llm.ai.memory import LongTermMemory
from fedot_llm.data.data import Dataset


class SupervisorAgent:
    def __init__(self, llm: BaseChatModel, memory: LongTermMemory, dataset: Dataset):
        self.llm = llm
        self.memory = memory
        self.dataset = dataset
        self.as_graph = self.create_graph()

    def create_graph(self):
        researcher_agent_tool = ResearcherAgent(llm=self.llm, memory=self.memory).as_tool
        automl_agent_tool = AutoMLAgent(llm=self.llm, dataset=self.dataset).as_tool
        return create_react_agent(self.llm, tools=[researcher_agent_tool, automl_agent_tool]).with_config(
            name='SupervisorAgent')
