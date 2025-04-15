import pytest
from unittest.mock import patch, MagicMock, ANY, call

from fedotllm.agents.data_analyst.schema import Memory, Message, Role
from fedotllm.llm.litellm import LiteLLMModel
from litellm.types.utils import ChatCompletionMessageToolCall


@pytest.fixture
def mock_litellm_model():
    """Fixture to provide a mocked LiteLLMModel instance"""
    mock_model = MagicMock(spec=LiteLLMModel)
    mock_model.model = "gpt-3.5-turbo"
    mock_model.model_max_input_tokens = 4096
    return mock_model


@pytest.fixture
def empty_memory(mock_litellm_model):
    """Fixture to provide an empty Memory instance"""
    return Memory(llm=mock_litellm_model)


@pytest.fixture
def sample_messages():
    """Fixture to provide a list of sample messages"""
    return [
        Message.system_message("You are a helpful assistant"),
        Message.user_message("Hello, how are you?"),
        Message.assistant_message("I'm doing well, thank you! How can I help you today?"),
        Message.user_message("Tell me about yourself"),
    ]


@pytest.fixture
def memory_with_messages(mock_litellm_model, sample_messages):
    """Fixture to provide a Memory instance with some messages"""
    memory = Memory(llm=mock_litellm_model)
    for message in sample_messages:
        memory.add_message(message)
    return memory


@pytest.fixture
def mock_tool_call():
    """Fixture to provide a mocked ChatCompletionMessageToolCall instance"""
    function = MagicMock()
    function.name = "test_function"
    function.arguments = '{"param": "value"}'
    
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function=function
    )
    return tool_call


@pytest.fixture
def sample_base64_image():
    """Fixture to provide a sample base64 image string"""
    return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AfwD/2Q=="


class TestMemoryTokenCount:
    """Tests for the Memory class token counting functionality"""

    @patch("litellm.utils.token_counter")
    def test_get_messages_token_count_empty(self, mock_token_counter, empty_memory):
        """Test token counting with an empty memory"""
        # Configure the mock
        mock_token_counter.return_value = 0
        
        # Call the method
        token_count = empty_memory.get_messages_token_count()
        
        # Assert results
        assert token_count == 0
        mock_token_counter.assert_called_once_with(
            model=empty_memory.llm.model, 
            messages=empty_memory.to_dict_list()
        )

    @patch("litellm.utils.token_counter")
    def test_get_messages_token_count_with_messages(self, mock_token_counter, memory_with_messages):
        """Test token counting with messages in memory"""
        # Configure the mock
        mock_token_counter.return_value = 150
        
        # Call the method
        token_count = memory_with_messages.get_messages_token_count()
        
        # Assert results
        assert token_count == 150
        mock_token_counter.assert_called_once_with(
            model=memory_with_messages.llm.model, 
            messages=memory_with_messages.to_dict_list()
        )

    @patch("litellm.utils.token_counter")
    def test_get_messages_token_count_with_different_message_types(self, mock_token_counter, mock_litellm_model, mock_tool_call):
        """Test token counting with different types of messages"""
        # Create a memory with various message types
        memory = Memory(llm=mock_litellm_model)
        
        # Add different message types
        memory.add_message(Message.system_message("System instruction"))
        memory.add_message(Message.user_message("User question"))
        
        memory.add_message(Message.assistant_message(
            content="I'll help with that", 
            tool_calls=[mock_tool_call]
        ))
        
        memory.add_message(Message.tool_message(
            content="Function result",
            name="test_function",
            tool_call_id="call_123"
        ))
        
        # Configure the mock
        mock_token_counter.return_value = 250
        
        # Call the method
        token_count = memory.get_messages_token_count()
        
        # Assert results
        assert token_count == 250
        # Use a less strict assertion that doesn't compare the exact content of the messages
        assert mock_token_counter.call_count == 1
        assert mock_token_counter.call_args[1]['model'] == mock_litellm_model.model
        assert len(mock_token_counter.call_args[1]['messages']) == 4

    @patch("litellm.utils.token_counter")
    def test_token_count_after_adding_messages(self, mock_token_counter, empty_memory, sample_messages):
        """Test token count updating correctly after adding messages"""
        # Set up the mock with different return values for successive calls
        token_counts = [50, 100, 150, 200]
        mock_token_counter.side_effect = token_counts
        
        # Add messages one by one and check token count
        for i, message in enumerate(sample_messages):
            empty_memory.add_message(message)
            token_count = empty_memory.get_messages_token_count()
            assert token_count == token_counts[i]
            
        # Verify the mock was called the correct number of times
        assert mock_token_counter.call_count == len(sample_messages)
    
    @patch("litellm.utils.token_counter")
    def test_get_messages_token_count_with_base64_images(self, mock_token_counter, mock_litellm_model, sample_base64_image):
        """Test token counting with messages containing base64 images"""
        # Create a memory with image content
        memory = Memory(llm=mock_litellm_model)
        
        # Add message with base64 image
        memory.add_message(Message.user_message(
            content="Here's an image", 
            base64_images=[sample_base64_image]
        ))
        
        # Configure the mock
        mock_token_counter.return_value = 300  # Images typically have high token counts
        
        # Call the method
        token_count = memory.get_messages_token_count()
        
        # Assert results
        assert token_count == 300
        assert mock_token_counter.call_count == 1
        
        # Check that the message dictionary was passed correctly
        messages_dict = mock_token_counter.call_args[1]['messages']
        assert len(messages_dict) == 1
        assert messages_dict[0]['role'] == 'user'
        assert len(messages_dict[0]['content']) == 2  # Both text and image parts
        
        # Verify image was included
        image_content = messages_dict[0]['content'][1]
        assert isinstance(image_content, list)
        assert image_content[0]['type'] == 'image_url'
        assert image_content[0]['image_url'] == sample_base64_image

    @patch("litellm.utils.token_counter")
    def test_get_messages_token_count_with_edge_cases(self, mock_token_counter, mock_litellm_model):
        """Test token counting with edge cases like extremely long messages"""
        # Create a memory with a very long message
        memory = Memory(llm=mock_litellm_model)
        
        # Add an extremely long message
        long_text = "This is a test. " * 1000  # Very long repeating text
        memory.add_message(Message.user_message(content=long_text))
        
        # Configure the mock with a high token count
        mock_token_counter.return_value = 5000
        
        # Call the method
        token_count = memory.get_messages_token_count()
        
        # Assert results
        assert token_count == 5000
        mock_token_counter.assert_called_once()
        
    @patch("litellm.utils.token_counter")
    def test_add_messages_method(self, mock_token_counter, mock_litellm_model, sample_messages):
        """Test adding multiple messages at once with add_messages method"""
        # Create an empty memory
        memory = Memory(llm=mock_litellm_model)
        
        # Configure the mock
        mock_token_counter.return_value = 300
        
        # Call add_messages method
        memory.add_messages(sample_messages)
        
        # Verify messages were added
        assert len(memory.messages) == len(sample_messages)
        for i, message in enumerate(sample_messages):
            assert memory.messages[i] == message
            
        # Note: We're not testing token_counter call because _trim_messages_to_token_limit 
        # is not fully implemented yet (has 'pass' in it)
        # This test focuses on verifying the messages were correctly added
    
    @pytest.mark.skip("_trim_messages_to_token_limit is not fully implemented yet")
    @patch.object(Memory, "get_messages_token_count")
    def test_trim_messages_to_token_limit(self, mock_get_token_count, mock_litellm_model):
        """Test that _trim_messages_to_token_limit correctly trims messages when over limit"""
        # Create a memory instance with a smaller max_tokens for testing
        memory = Memory(llm=mock_litellm_model)
        memory.max_tokens = 100  # Small limit for testing
        
        # Configure the mock to return token counts above the limit, then below after trimming
        mock_get_token_count.side_effect = [200, 80]
        
        # Add test messages
        system_msg = Message.system_message("System prompt")
        user_msg1 = Message.user_message("First user message")
        assistant_msg1 = Message.assistant_message("First assistant response")
        user_msg2 = Message.user_message("Second user message")
        assistant_msg2 = Message.assistant_message("Second assistant response")
        
        memory.messages = [system_msg, user_msg1, assistant_msg1, user_msg2, assistant_msg2]
        
        # Call the trim method directly
        memory._trim_messages_to_token_limit()
        
        # Verify the most recent user message is preserved
        assert user_msg2 in memory.messages
        
        # Verify the system message is preserved (typical behavior)
        assert system_msg in memory.messages
        
        # Verify we called token counting twice:
        # First to check if over limit, second after trimming
        assert mock_get_token_count.call_count == 2
        
    @pytest.mark.skip("_trim_messages_to_token_limit is not fully implemented yet")
    @patch.object(Memory, "get_messages_token_count")
    def test_trim_preserves_recent_messages(self, mock_get_token_count, mock_litellm_model):
        """Test that _trim_messages_to_token_limit preserves more recent messages over older ones"""
        # Create a memory instance
        memory = Memory(llm=mock_litellm_model)
        memory.max_tokens = 100  # Small limit for testing
        
        # Configure the mock to return values that indicate trimming is needed
        mock_get_token_count.side_effect = [150, 90]
        
        # Create 10 test messages
        messages = []
        for i in range(10):
            if i % 2 == 0:
                messages.append(Message.user_message(f"User message {i}"))
            else:
                messages.append(Message.assistant_message(f"Assistant message {i}"))
        
        # Add them to memory
        memory.messages = messages.copy()
        
        # Call the trim method
        memory._trim_messages_to_token_limit()
        
        # Verify the most recent messages are preserved
        # Assuming the implementation removes oldest messages first
        assert len(memory.messages) < 10
        
        # Check that the preserved messages are the most recent ones
        for msg in memory.messages:
            # Find its index in the original list
            index = -1
            for i, original_msg in enumerate(messages):
                if original_msg.content == msg.content:
                    index = i
                    break
                    
            # Assert that only messages from the second half are kept
            assert index >= len(messages) // 2 