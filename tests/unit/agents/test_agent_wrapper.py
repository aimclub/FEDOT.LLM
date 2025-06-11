import pytest
from unittest.mock import MagicMock, patch, ANY, partial, call # Added 'call'

from fedotllm.agents.agent_wrapper import AgentWrapper
from fedotllm.agents.base import Agent # For spec in mocks
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.automl.state import AutoMLAgentState
# Mocked langgraph components are handled by @patch decorators

# Fixtures
@pytest.fixture
def mock_researcher_agent(mocker):
    agent = MagicMock(spec=ResearcherAgent)
    return agent

@pytest.fixture
def mock_automl_agent(mocker):
    agent = MagicMock(spec=AutoMLAgent)
    return agent

@pytest.fixture
def mock_generic_agent(mocker):
    agent = MagicMock(spec=Agent)
    return agent


# Tests for __init__
def test_agent_wrapper_init(mock_researcher_agent):
    wrapper = AgentWrapper(agent=mock_researcher_agent)
    assert wrapper.agent == mock_researcher_agent


# Tests for create_graph
@patch('fedotllm.agents.agent_wrapper.END', new_callable=lambda: "END_NODE")
@patch('fedotllm.agents.agent_wrapper.START', new_callable=lambda: "START_NODE")
@patch('fedotllm.agents.agent_wrapper.run_exit_state_connector')
@patch('fedotllm.agents.agent_wrapper.run_entry_state_connector')
@patch('fedotllm.agents.agent_wrapper.StateGraph')
def test_create_graph_with_researcher_agent(
    mock_state_graph_constructor, 
    mock_run_entry_connector, 
    mock_run_exit_connector, 
    mock_start_node_str, # Now these are strings from new_callable
    mock_end_node_str,
    mock_researcher_agent
):
    mock_agent_graph = MagicMock(name="MockedAgentInternalGraph")
    mock_researcher_agent.create_graph.return_value = mock_agent_graph
    
    mock_workflow_instance = MagicMock(name="StateGraphInstance")
    mock_state_graph_constructor.return_value = mock_workflow_instance
    
    wrapper = AgentWrapper(agent=mock_researcher_agent) # Removed 'name' argument
    compiled_graph = wrapper.create_graph()

    mock_state_graph_constructor.assert_called_once_with(ResearcherAgentState)
    
    # Check add_node calls
    # For partials, direct comparison can be tricky. Check the function part and if 'agent' kwarg is correct.
    add_node_calls = mock_workflow_instance.add_node.call_args_list
    
    # Entry connector node
    assert add_node_calls[0][0][0] == "run_entry_state_connector"
    assert isinstance(add_node_calls[0][0][1], partial)
    assert add_node_calls[0][0][1].func == mock_run_entry_connector
    assert add_node_calls[0][0][1].keywords.get("agent") == mock_researcher_agent
    
    # Agent node
    assert add_node_calls[1] == call("agent", mock_agent_graph)
    
    # Exit connector node
    assert add_node_calls[2][0][0] == "run_exit_state_connector"
    assert isinstance(add_node_calls[2][0][1], partial)
    assert add_node_calls[2][0][1].func == mock_run_exit_connector
    assert add_node_calls[2][0][1].keywords.get("agent") == mock_researcher_agent

    # Check add_edge calls
    mock_workflow_instance.add_edge.assert_has_calls([
        call(mock_start_node_str, "run_entry_state_connector"),
        call("run_entry_state_connector", "agent"),
        call("agent", "run_exit_state_connector"),
        call("run_exit_state_connector", mock_end_node_str)
    ], any_order=False) # Order matters for edges

    mock_workflow_instance.compile.assert_called_once()
    assert compiled_graph == mock_workflow_instance.compile.return_value


@patch('fedotllm.agents.agent_wrapper.END', new_callable=lambda: "END_NODE")
@patch('fedotllm.agents.agent_wrapper.START', new_callable=lambda: "START_NODE")
@patch('fedotllm.agents.agent_wrapper.run_exit_state_connector')
@patch('fedotllm.agents.agent_wrapper.run_entry_state_connector')
@patch('fedotllm.agents.agent_wrapper.StateGraph')
def test_create_graph_with_automl_agent(
    mock_state_graph_constructor, 
    mock_run_entry_connector, 
    mock_run_exit_connector, 
    mock_start_node_str, 
    mock_end_node_str,
    mock_automl_agent
):
    mock_agent_graph = MagicMock(name="MockedAgentInternalGraph")
    mock_automl_agent.create_graph.return_value = mock_agent_graph
    
    mock_workflow_instance = MagicMock(name="StateGraphInstance")
    mock_state_graph_constructor.return_value = mock_workflow_instance
    
    wrapper = AgentWrapper(agent=mock_automl_agent)
    compiled_graph = wrapper.create_graph()

    mock_state_graph_constructor.assert_called_once_with(AutoMLAgentState)
    
    add_node_calls = mock_workflow_instance.add_node.call_args_list
    assert add_node_calls[0][0][0] == "run_entry_state_connector"
    assert isinstance(add_node_calls[0][0][1], partial)
    assert add_node_calls[0][0][1].func == mock_run_entry_connector
    assert add_node_calls[0][0][1].keywords.get("agent") == mock_automl_agent
    
    assert add_node_calls[1] == call("agent", mock_agent_graph)
    
    assert add_node_calls[2][0][0] == "run_exit_state_connector"
    assert isinstance(add_node_calls[2][0][1], partial)
    assert add_node_calls[2][0][1].func == mock_run_exit_connector
    assert add_node_calls[2][0][1].keywords.get("agent") == mock_automl_agent

    mock_workflow_instance.add_edge.assert_has_calls([
        call(mock_start_node_str, "run_entry_state_connector"),
        call("run_entry_state_connector", "agent"),
        call("agent", "run_exit_state_connector"),
        call("run_exit_state_connector", mock_end_node_str)
    ], any_order=False)

    mock_workflow_instance.compile.assert_called_once()
    assert compiled_graph == mock_workflow_instance.compile.return_value


def test_create_graph_unsupported_agent(mock_generic_agent):
    wrapper = AgentWrapper(agent=mock_generic_agent)
    with pytest.raises(ValueError, match="Not supported agent in AgentWrapper."):
        wrapper.create_graph()
