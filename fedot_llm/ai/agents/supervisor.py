import yaml
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph import MessagesState

from fedot_llm.ai.agents.automl.automl import AutoMLAgent
from fedot_llm.ai.agents.researcher.researcher import ResearcherAgent
from fedot_llm.ai.memory import LongTermMemory
from fedot_llm.data.data import Dataset


class SupervisorState(MessagesState):
    """The state of the agent."""
    next: str


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "You are a supervisor. "
               "You are given a conversation between a user and an agents. "
               "You need to decide who should act next. Or should we FINISH? "
               "You need to return the tool that should act next. "
               "Or return FINISH if the task is solved.")
])


class SupervisorAgent:
    def __init__(self, llm: BaseChatModel, memory: LongTermMemory, dataset: Dataset):
        self.llm = llm
        self.memory = memory
        self.dataset = dataset
        self.as_graph = self.create_graph()

    def create_graph(self):
        workflow = StateGraph(SupervisorState)

        researcher_agent = ResearcherAgent(llm=self.llm, memory=self.memory)
        automl_agent = AutoMLAgent(llm=self.llm, dataset=self.dataset)

        @tool("FINISH")
        def finish_tool():
            """Finish the conversation"""
            pass

        def supervisor_chain(state: SupervisorState):
            messages = state["messages"]
            response = self.llm.bind_tools([researcher_agent.as_tool, automl_agent.as_tool, finish_tool]).invoke(
                messages)
            if isinstance(response, AIMessage) and hasattr(response, "tool_calls"):
                if not response.tool_calls:
                    return {"next": "FINISH"}
                else:
                    next = response.tool_calls[0]["name"]
                    call_message = {"calls": {response.tool_calls[0]["name"]: {"args": response.tool_calls[0]["args"]}}}
                    return {"messages": [*messages,
                                         AIMessage(content=yaml.dump(call_message))],
                            "next": next}

        agent_by_name = {
            "ResearcherAgent": researcher_agent.as_graph,
            "AutoMLAgent": automl_agent.as_graph
        }

        def should_continue(state: SupervisorState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and state["next"] in agent_by_name.keys():
                return state["next"]
            else:
                return "end"

        workflow.set_entry_point("SupervisorAgent")
        workflow.add_node("SupervisorAgent", supervisor_chain)

        for agent_name, agent_graph in agent_by_name.items():
            workflow.add_node(agent_name, agent_graph)

        workflow.add_conditional_edges(
            "SupervisorAgent",
            should_continue,
            {
                "end": END,
                **{agent_name: agent_name for agent_name in agent_by_name.keys()}
            }
        )

        workflow.add_edge("ResearcherAgent", "SupervisorAgent")
        workflow.add_edge("AutoMLAgent", "SupervisorAgent")

        return workflow.compile()
