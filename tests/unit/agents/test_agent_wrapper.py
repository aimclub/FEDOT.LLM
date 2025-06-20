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

# Tests for run_entry_state_connector
def test_run_entry_state_connector_researcher(mock_researcher_agent):
    from fedotllm.agents.agent_wrapper import run_entry_state_connector
    
    # Simulate a state dictionary
    mock_message_content = "What is the capital of France?"
    mock_message = MagicMock()
    mock_message.content = mock_message_content
    
    initial_state = {
        "messages": [MagicMock(), mock_message] # List of messages, last one is relevant
    }
    
    # Act
    updated_state = run_entry_state_connector(initial_state, agent=mock_researcher_agent)
    
    # Assert
    assert updated_state["question"] == mock_message_content
    assert updated_state["messages"] == initial_state["messages"] # Ensure other parts of state are preserved

def test_run_entry_state_connector_automl(mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_entry_state_connector
    
    mock_message_content = "Generate a classification model for this data."
    mock_message = MagicMock()
    mock_message.content = mock_message_content
    
    initial_state = {
        "messages": [mock_message] 
    }
    
    updated_state = run_entry_state_connector(initial_state, agent=mock_automl_agent)
    
    assert updated_state["description"] == mock_message_content
    assert updated_state["messages"] == initial_state["messages"]

def test_run_entry_state_connector_researcher_empty_messages(mock_researcher_agent):
    from fedotllm.agents.agent_wrapper import run_entry_state_connector
    # Test with empty messages list - expecting IndexError
    initial_state = {"messages": []}
    with pytest.raises(IndexError):
        run_entry_state_connector(initial_state, agent=mock_researcher_agent)

def test_run_entry_state_connector_automl_empty_messages(mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_entry_state_connector
    initial_state = {"messages": []}
    with pytest.raises(IndexError):
        run_entry_state_connector(initial_state, agent=mock_automl_agent)

# Tests for run_exit_state_connector
@patch('fedotllm.agents.agent_wrapper.HumanMessage', autospec=True)
def test_run_exit_state_connector_researcher(mock_human_message_constructor, mock_researcher_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    
    mock_answer = "Paris is the capital of France."
    initial_state = {
        "answer": mock_answer,
        "other_data": "some_value" 
    }
    
    # Expected HumanMessage instance
    mock_human_message_instance = MagicMock(name="HumanMessageInstance")
    mock_human_message_constructor.return_value = mock_human_message_instance
    
    updated_state = run_exit_state_connector(initial_state, agent=mock_researcher_agent)
    
    mock_human_message_constructor.assert_called_once_with(content=mock_answer, name="ResearcherAgent")
    assert updated_state["messages"] == [mock_human_message_instance]
    assert updated_state["answer"] == mock_answer # Ensure original state parts are preserved
    assert updated_state["other_data"] == "some_value"


@patch('fedotllm.agents.agent_wrapper.HumanMessage', autospec=True)
def test_run_exit_state_connector_automl(mock_human_message_constructor, mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    
    mock_code = "print('hello world')"
    initial_state = {
        "solutions": [
            {"code": "print('old solution')"},
            {"code": mock_code} # Last solution is relevant
        ],
        "other_data": "automl_value"
    }
    
    mock_human_message_instance = MagicMock(name="HumanMessageInstance")
    mock_human_message_constructor.return_value = mock_human_message_instance
    
    updated_state = run_exit_state_connector(initial_state, agent=mock_automl_agent)
    
    mock_human_message_constructor.assert_called_once_with(content=mock_code, name="AutoMLAgent")
    assert updated_state["messages"] == [mock_human_message_instance]
    assert updated_state["solutions"] == initial_state["solutions"]
    assert updated_state["other_data"] == "automl_value"

def test_run_exit_state_connector_researcher_missing_answer(mock_researcher_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    initial_state = {"other_data": "value"} # 'answer' key is missing
    with pytest.raises(KeyError, match="'answer'"):
        run_exit_state_connector(initial_state, agent=mock_researcher_agent)

def test_run_exit_state_connector_automl_missing_solutions(mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    initial_state = {"other_data": "value"} # 'solutions' key is missing
    with pytest.raises(KeyError, match="'solutions'"):
        run_exit_state_connector(initial_state, agent=mock_automl_agent)

def test_run_exit_state_connector_automl_empty_solutions(mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    initial_state = {"solutions": [], "other_data": "value"} # 'solutions' is empty
    with pytest.raises(IndexError): # Expecting access to solutions[-1] to fail
        run_exit_state_connector(initial_state, agent=mock_automl_agent)

def test_run_exit_state_connector_automl_solution_missing_code(mock_automl_agent):
    from fedotllm.agents.agent_wrapper import run_exit_state_connector
    initial_state = {
        "solutions": [{"description": "solution without code"}], 
        "other_data": "value"
    }
    with pytest.raises(KeyError, match="'code'"): # Expecting access to solution['code'] to fail
        run_exit_state_connector(initial_state, agent=mock_automl_agent)
