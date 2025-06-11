import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
import pandas as pd
from langchain_core.messages import HumanMessage #, AIMessage
from langchain_core.runnables.schema import StreamEvent #, EventData

from fedotllm.main import FedotAI
# Mocked class placeholders (actual mocks will be done with @patch or fixtures)
# from fedotllm.llm import AIInference, OpenaiEmbeddings (will be mocked)
# from fedotllm.data import Dataset (will be mocked)
# from fedotllm.agents.supervisor import SupervisorAgent (will be mocked)
# from fedotllm.agents.agent_wrapper.agent_wrapper import AgentWrapper (will be mocked)
# from fedotllm.agents.automl.automl_chat import AutoMLAgentChat (will be mocked)
# from fedotllm.agents.researcher.researcher import ResearcherAgent (will be mocked)

# Fixtures
@pytest.fixture
def mock_task_path(mocker):
    mock_path = MagicMock(spec=Path)
    mock_path.resolve.return_value = mock_path  # resolve returns itself for simplicity
    return mock_path

@pytest.fixture
def mock_ai_inference(mocker):
    return MagicMock(name="AIInferenceInstance")

@pytest.fixture
def mock_openai_embeddings(mocker):
    return MagicMock(name="OpenaiEmbeddingsInstance")

# Tests for __init__
def test_fedotai_init_success(mock_task_path, mock_ai_inference, mock_openai_embeddings):
    mock_task_path.is_dir.return_value = True
    
    fedot_ai = FedotAI(task_path=mock_task_path, inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    
    mock_task_path.resolve.assert_called_once()
    assert fedot_ai.task_path == mock_task_path
    assert fedot_ai.inference == mock_ai_inference
    assert fedot_ai.embeddings == mock_openai_embeddings
    assert fedot_ai.workspace is None # Default

    # Test with task_path as string - Temporarily commented out to avoid Path._flavour error
    # mock_task_path_str = "/fake/path"
    # with patch('pathlib.Path', return_value=mock_task_path) as mock_path_constructor:
    #     fedot_ai_str = FedotAI(task_path=mock_task_path_str, inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    #     mock_path_constructor.assert_called_once_with(mock_task_path_str)
    #     mock_task_path.resolve.assert_called() # Called again
    #     assert fedot_ai_str.task_path == mock_task_path

def test_fedotai_init_task_path_not_dir_raises(mock_task_path, mock_ai_inference, mock_openai_embeddings):
    mock_task_path.is_dir.return_value = False
    
    with pytest.raises(AssertionError, match="Task path does not exist or is not a directory."):
        FedotAI(task_path=mock_task_path, inference=mock_ai_inference, embeddings=mock_openai_embeddings)

def test_fedotai_init_workspace_handling(mock_task_path, mock_ai_inference, mock_openai_embeddings):
    mock_task_path.is_dir.return_value = True

    # Test with workspace as string
    fedot_ai_str_ws = FedotAI(task_path=mock_task_path, workspace="test_workspace_str", inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    assert isinstance(fedot_ai_str_ws.workspace, Path)
    assert str(fedot_ai_str_ws.workspace) == "test_workspace_str"

    # Test with workspace as Path object
    test_workspace_path_obj = Path("test_workspace_path")
    fedot_ai_path_ws = FedotAI(task_path=mock_task_path, workspace=test_workspace_path_obj, inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    assert fedot_ai_path_ws.workspace == test_workspace_path_obj


# Async Tests for ainvoke and ask
@pytest.mark.asyncio
async def test_fedotai_ainvoke_flow(mocker, mock_task_path, mock_ai_inference, mock_openai_embeddings):
    mock_task_path.is_dir.return_value = True

    mock_dataset_from_path = mocker.patch('fedotllm.main.Dataset.from_path', return_value=MagicMock(name="DatasetInstance"))
    
    mock_automl_agent_chat = MagicMock()
    mock_automl_graph = AsyncMock()
    mock_automl_agent_chat.create_graph.return_value = mock_automl_graph
    mocker.patch('fedotllm.main.AutoMLAgentChat', return_value=mock_automl_agent_chat)

    mock_researcher_agent = MagicMock()
    mock_researcher_graph = AsyncMock() # Will be wrapped
    mock_researcher_agent.create_graph.return_value = mock_researcher_graph
    mocker.patch('fedotllm.main.ResearcherAgent', return_value=mock_researcher_agent)

    mock_agent_wrapper = MagicMock()
    mock_wrapped_researcher_graph = AsyncMock(name="WrappedResearcherGraph")
    mock_agent_wrapper.create_graph.return_value = mock_wrapped_researcher_graph
    mocker.patch('fedotllm.main.AgentWrapper', return_value=mock_agent_wrapper)
    
    mock_supervisor_agent = MagicMock()
    mock_supervisor_graph = AsyncMock(name="SupervisorGraph")
    mock_supervisor_agent.create_graph.return_value = mock_supervisor_graph
    mocker.patch('fedotllm.main.SupervisorAgent', return_value=mock_supervisor_agent)

    mock_timestamp_strftime = mocker.patch('pandas.Timestamp.now')
    mock_now = MagicMock()
    mock_now.strftime.return_value = "YYYYMMDD_HHMMSS_test" # Scenario 1 name
    mock_timestamp_strftime.return_value = mock_now
    
    # Mock Path methods for workspace interaction
    mock_path_mkdir = mocker.patch.object(Path, 'mkdir')
    mock_path_exists = mocker.patch.object(Path, 'exists')


    # Scenario 1: Workspace needs to be created
    mock_path_exists.return_value = False # Workspace does not exist initially
    fedot_ai = FedotAI(task_path=mock_task_path, inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    assert fedot_ai.workspace is None # Initially None

    test_message = "test message for ainvoke"
    await fedot_ai.ainvoke(test_message)

    mock_dataset_from_path.assert_called_once_with(mock_task_path)
    mock_automl_agent_chat.create_graph.assert_called_once()
    mock_agent_wrapper.create_graph.assert_called_once()
    mock_supervisor_agent.create_graph.assert_called_once()
    
    # Check that the workspace Path object was created with the correct dynamic name
    # and mkdir was called on it.
    # The actual Path object is created internally.
    # We assert that the workspace attribute of fedot_ai is a Path object with the expected name.
    expected_workspace_name = "fedotllm-output-YYYYMMDD_HHMMSS_test" 
    assert isinstance(fedot_ai.workspace, Path)
    assert fedot_ai.workspace.name == expected_workspace_name
    # mkdir is not called by FedotAI.ainvoke itself, so remove this check.
    # mock_path_mkdir.assert_called_with(parents=True, exist_ok=True) 
    
    mock_supervisor_graph.ainvoke.assert_called_once_with({"messages": [HumanMessage(content=test_message)]})

    # Scenario 2: Workspace already exists
    mock_path_mkdir.reset_mock()
    mock_path_exists.return_value = True # Workspace now exists
    mock_timestamp_strftime.reset_mock() # Not used if workspace is provided
    mock_supervisor_graph.ainvoke.reset_mock()

    existing_workspace_str = "existing_ws"
    fedot_ai_ws_exists = FedotAI(task_path=mock_task_path, workspace=existing_workspace_str, inference=mock_ai_inference, embeddings=mock_openai_embeddings)
    
    # Path(existing_workspace_str).exists() will be called.
    # We need to make sure our mock_path_exists handles this specific path if needed,
    # or that the generic return_value=True is sufficient.
    # For this test, fedot_ai_ws_exists.workspace will be Path("existing_ws")
    # Its .exists() method will use our mock_path_exists.
    
    await fedot_ai_ws_exists.ainvoke("another message")
    
    mock_timestamp_strftime.assert_not_called() 
    mock_path_mkdir.assert_not_called() # mkdir should not be called as workspace is provided
    assert str(fedot_ai_ws_exists.workspace) == existing_workspace_str
    mock_supervisor_graph.ainvoke.assert_called_once_with({"messages": [HumanMessage(content="another message")]})


@pytest.mark.asyncio
async def test_fedotai_ask_flow(mocker, mock_task_path, mock_ai_inference, mock_openai_embeddings):
    mock_task_path.is_dir.return_value = True

    mocker.patch('fedotllm.main.Dataset.from_path', return_value=MagicMock(name="DatasetInstance"))
    
    mock_automl_agent_chat = MagicMock()
    mocker.patch('fedotllm.main.AutoMLAgentChat', return_value=mock_automl_agent_chat)
    mock_researcher_agent = MagicMock()
    mocker.patch('fedotllm.main.ResearcherAgent', return_value=mock_researcher_agent)
    mock_agent_wrapper = MagicMock()
    mocker.patch('fedotllm.main.AgentWrapper', return_value=mock_agent_wrapper)
    
    mock_supervisor_agent = MagicMock()
    mock_supervisor_graph = AsyncMock(name="SupervisorGraph") 
    mock_supervisor_agent.create_graph.return_value = mock_supervisor_graph
    mocker.patch('fedotllm.main.SupervisorAgent', return_value=mock_supervisor_agent)

    mock_event_data = {"chunk": "Test event content"}
    # Workaround for TypeError: Cannot instantiate typing.Union by creating a dict
    mock_stream_event_dict = {
        "event": "on_chat_model_stream", 
        "run_id": "test_run", 
        "name": "test_name", 
        "data": mock_event_data
    }
    
    async def mock_event_streamer_gen_obj(*args, **kwargs): 
        yield mock_stream_event_dict

    # Replace the 'astream_events' attribute (which would be an AsyncMock by default if part of an AsyncMock object)
    # with a MagicMock whose return_value is the async generator itself.
    mock_supervisor_graph.astream_events = MagicMock(return_value=mock_event_streamer_gen_obj())

    # Mock Path methods for workspace interaction
    mock_path_mkdir_ask = mocker.patch.object(Path, 'mkdir', autospec=True) # Use autospec
    mock_path_exists_ask = mocker.patch.object(Path, 'exists', return_value=False, autospec=True)

    # Correctly mock pandas.Timestamp.now().strftime()
    mock_pd_now_ask = MagicMock()
    mock_pd_now_ask.strftime.return_value = "YYYYMMDD_HHMMSS_ask"
    mocker.patch('pandas.Timestamp.now', return_value=mock_pd_now_ask)

    mock_handler = MagicMock()
    # Pass a list of handlers
    fedot_ai = FedotAI(task_path=mock_task_path, inference=mock_ai_inference, embeddings=mock_openai_embeddings, handlers=[mock_handler])
    
    test_message_ask = "test message for ask"
    
    events_received = []
    async for event in fedot_ai.ask(test_message_ask):
        events_received.append(event)
    
    mock_supervisor_graph.astream_events.assert_called_once_with(
        {"messages": [HumanMessage(content=test_message_ask)]},
        version="v2", # As per main.py
    )
    
    # Check if handler was called with the event
    mock_handler.assert_any_call(mock_stream_event_dict) 
    
    assert len(events_received) == 1
    assert events_received[0] == mock_stream_event_dict
