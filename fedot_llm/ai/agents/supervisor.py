from langchain.chat_models.base import BaseChatModel
from fedot_llm.ai.agents.automl import AutoMLAgent
from fedot_llm.ai.agents.researcher.researcher import ResearcherAgent
from fedot_llm.ai.memory import LongTermMemory
from fedot_llm.data.data import Dataset
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import json

class SupervisorState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    args: dict
    
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
        
        researcher_agent_tool = ResearcherAgent(llm=self.llm, memory=self.memory).as_tool
        automl_agent_tool = AutoMLAgent(llm=self.llm, dataset=self.dataset).as_tool
        
        @tool("FINISH")
        def finish_tool():
            """Finish the conversation"""
            pass
        
        def supervisor_chain(state: SupervisorState):
            messages = state["messages"]
            next = state.get("next", "")
            args = state.get("args", {})
            response = self.llm.bind_tools([researcher_agent_tool, automl_agent_tool, finish_tool]).invoke(messages)
            if isinstance(response, AIMessage) and hasattr(response, "tool_calls"):
                if not response.tool_calls:
                    next = "FINISH"
                else:
                    next = response.tool_calls[0]["name"]
                    args = response.tool_calls[0]["args"]
                    return {"messages": AIMessage(content=f"Calling {next} for help. With args: {str(args)}"), 
                            "next": next, 
                            "args": args}
        
        agent_by_name = {
            "ResearcherAgent": researcher_agent_tool,
            "AutoMLAgent": automl_agent_tool
        }
        
        def agent_call(state: SupervisorState):
            outputs = []
            last_message = state['messages'][-1]
            next_agent = state["next"]
            args = state["args"]
            if isinstance(last_message, AIMessage) and next_agent in agent_by_name.keys():
                tool_result = agent_by_name[next_agent].invoke(
                    args
                )
                outputs.append(
                    HumanMessage(
                    content=str(tool_result),
                    name=next_agent,
                )
            )
            return {"messages": outputs, "next": "Supervisor", "args": {}}
        
        def should_continue(state: SupervisorState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and state["next"] in agent_by_name.keys():
                return "continue"
            else:
                return "end"
            
        workflow.set_entry_point("Supervisor")
        workflow.add_node("Supervisor", supervisor_chain)
        workflow.add_node("AgentCall", agent_call)

        workflow.add_conditional_edges(
            "Supervisor",
            should_continue,
            {
                "end": END,
                "continue": "AgentCall"
            }
        )
        
        workflow.add_edge("AgentCall", "Supervisor")
        return workflow.compile()
        
