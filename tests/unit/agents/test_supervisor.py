import pytest
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from functools import partial

from fedotllm.agents.supervisor import SupervisorAgent, choose_next, NextAgent, SupervisorState, ChooseNext
from fedotllm.llm import AIInference
from fedotllm.agents.base import FedotLLMAgentState # For SupervisorState
from langchain_core.messages import HumanMessage, SystemMessage 
from langgraph.types import Command
# from langgraph.graph import StateGraph, START, END # Mocked

# Fixtures
@pytest.fixture
def mock_inference(mocker):
    return MagicMock(spec=AIInference)

@pytest.fixture
def mock_automl_agent_runnable(mocker):
    runnable = MagicMock(name="AutoMLAgentRunnable")
    # If the runnable is directly invoked or its methods are, mock them as needed
    # For example, if it's used in graph.add_node(..., runnable_instance),
    # it might need to be callable or have specific attributes.
    return runnable

@pytest.fixture
def mock_researcher_agent_runnable(mocker):
    runnable = MagicMock(name="ResearcherAgentRunnable")
    return runnable


# Tests for choose_next function
@patch('fedotllm.agents.supervisor.choose_next_prompt')
def test_choose_next_routes_to_automl(mock_choose_next_prompt_func, mock_inference): # Renamed mock
    mock_choose_next_prompt_func.return_value = "Test Prompt for AutoML"
    mock_inference.create.return_value = ChooseNext(next=NextAgent.AUTOML)
    
    state = SupervisorState(messages=[HumanMessage(content="build a model")], next=None)
    
    command = choose_next(state, mock_inference)
    
    # HumanMessage default name might be None or 'human'. The formatting in supervisor.py uses m.name.
    # If HumanMessage.name is None, it results in "None: content".
    # If it's not set and Langchain gives a default like "human", it'd be "human: content".
    # Let's assume it's "None" for now if not specified, based on typical BaseMessage behavior.
    expected_formatted_message = "None: build a model" 
    mock_choose_next_prompt_func.assert_called_once_with(expected_formatted_message)
    mock_inference.create.assert_called_once_with("Test Prompt for AutoML", response_model=ChooseNext)
    assert command == Command(goto=NextAgent.AUTOML)

@patch('fedotllm.agents.supervisor.choose_next_prompt')
def test_choose_next_routes_to_researcher(mock_choose_next_prompt_func, mock_inference):
    mock_choose_next_prompt_func.return_value = "Test Prompt for Researcher"
    mock_inference.create.return_value = ChooseNext(next=NextAgent.RESEARCHER)
    
    state = SupervisorState(messages=[HumanMessage(content="what is fedot?")], next=None)
    
    command = choose_next(state, mock_inference)
    
    expected_formatted_message = "None: what is fedot?"
    mock_choose_next_prompt_func.assert_called_once_with(expected_formatted_message)
    mock_inference.create.assert_called_once_with("Test Prompt for Researcher", response_model=ChooseNext)
    assert command == Command(goto=NextAgent.RESEARCHER)

@patch('fedotllm.agents.supervisor.choose_next_prompt')
def test_choose_next_routes_to_finish(mock_choose_next_prompt_func, mock_inference):
    mock_choose_next_prompt_func.return_value = "Test Prompt for Finish"
    mock_inference.create.return_value = ChooseNext(next=NextAgent.FINISH)
    
    state = SupervisorState(messages=[HumanMessage(content="thank you")], next=None)
    
    command = choose_next(state, mock_inference)
    
    expected_formatted_message = "None: thank you"
    mock_choose_next_prompt_func.assert_called_once_with(expected_formatted_message)
    mock_inference.create.assert_called_once_with("Test Prompt for Finish", response_model=ChooseNext)
    assert command == Command(goto=NextAgent.FINISH)

@patch('fedotllm.agents.supervisor.choose_next_prompt')
def test_choose_next_formats_multiple_messages(mock_choose_next_prompt_func, mock_inference):
    messages = [
        HumanMessage(name="User", content="Hello"), 
        SystemMessage(content="Hi there! I am an AI model.") # SystemMessage doesn't take 'name'
    ]
    state = SupervisorState(messages=messages, next=None)
    mock_inference.create.return_value = ChooseNext(next=NextAgent.FINISH) # Arbitrary next agent

    choose_next(state, mock_inference)
    
    # The choose_next_prompt function in supervisor.py formats messages.
    expected_formatted_messages = "User: Hello\nNone: Hi there! I am an AI model."
    mock_choose_next_prompt_func.assert_called_once_with(expected_formatted_messages)


# Tests for SupervisorAgent class
@patch('fedotllm.agents.supervisor.END', new_callable=lambda: "END_NODE_SUPERVISOR") # Unique names to avoid collision
@patch('fedotllm.agents.supervisor.START', new_callable=lambda: "START_NODE_SUPERVISOR")
@patch('fedotllm.agents.supervisor.StateGraph')
class TestSupervisorAgent:

    def test_supervisor_agent_init(self, mock_state_graph_ignored, mock_start_ignored, mock_end_ignored, 
                                   mock_inference, mock_automl_agent_runnable, mock_researcher_agent_runnable):
        # Patches are active but not directly checked in __init__
        agent = SupervisorAgent(
            inference=mock_inference, 
            automl_agent=mock_automl_agent_runnable, 
            researcher_agent=mock_researcher_agent_runnable
        )
        assert agent.inference == mock_inference
        assert agent.automl_agent == mock_automl_agent_runnable
        assert agent.researcher_agent == mock_researcher_agent_runnable

    def test_supervisor_agent_create_graph(self, mock_state_graph_constructor, mock_start_node_str, mock_end_node_str,
                                           mock_inference, mock_automl_agent_runnable, mock_researcher_agent_runnable):
        
        mock_workflow_instance = MagicMock(name="StateGraphInstance")
        mock_state_graph_constructor.return_value = mock_workflow_instance
        
        # Mock the return of .compile().with_config(...)
        mock_compiled_graph_final = MagicMock(name="FinalCompiledGraphWithConfig")
        mock_compiled_workflow = MagicMock(name="InitialCompiledGraph")
        mock_workflow_instance.compile.return_value = mock_compiled_workflow
        mock_compiled_workflow.with_config.return_value = mock_compiled_graph_final

        agent = SupervisorAgent(
            inference=mock_inference, 
            automl_agent=mock_automl_agent_runnable, 
            researcher_agent=mock_researcher_agent_runnable
        )
        
        compiled_graph = agent.create_graph()

        mock_state_graph_constructor.assert_called_once_with(SupervisorState)
        
        # Check add_node calls
        add_node_calls = mock_workflow_instance.add_node.call_args_list
        
        # choose_next node
        assert add_node_calls[0][0][0] == "choose_next"
        assert isinstance(add_node_calls[0][0][1], partial)
        assert add_node_calls[0][0][1].func == choose_next # Check it's the actual function
        assert add_node_calls[0][0][1].keywords.get("inference") == mock_inference
        
        # Agent nodes
        assert call("researcher", mock_researcher_agent_runnable) in add_node_calls
        assert call("automl", mock_automl_agent_runnable) in add_node_calls
        
        # finish_execution node (ANY because it's a direct function reference from supervisor.py)
        # We can be more specific if we import finish_execution, but ANY is fine for now
        found_finish_node = False
        for node_call in add_node_calls:
            if node_call[0][0] == "finish":
                found_finish_node = True
                assert callable(node_call[0][1]) # Check that it's a callable (the finish_execution function)
                break
        assert found_finish_node, "finish node not found"

        # Check other direct edges
        mock_workflow_instance.add_edge.assert_has_calls([
            call(mock_start_node_str, "choose_next"),
            call("researcher", "choose_next"), # Loop back after researcher
            call("automl", "finish"),         # AutoML goes to finish
            call("finish", mock_end_node_str)
        ], any_order=True) # Use any_order=True because conditional_edges might affect order relative to these

        mock_workflow_instance.compile.assert_called_once()
        mock_compiled_workflow.with_config.assert_called_once_with(config={"run_name": "SupervisorAgent"})
        assert compiled_graph == mock_compiled_graph_final
