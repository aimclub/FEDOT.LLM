from unittest.mock import patch, MagicMock
import pytest
import os
from pydantic import BaseModel, Field, ValidationError
from fedotllm.llm import AIInference
from tenacity import wait_none

class UserModel(BaseModel):
    """Test model for structured response testing"""
    name: str = Field(..., description="User name")
    age: int = Field(..., description="User age", ge=0, le=120)
    email: str = Field(..., description="User email")
    active: bool = Field(default=True, description="User active status")

# Fixtures
@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch('fedotllm.llm.get_settings') as mock:
        mock_settings_obj = MagicMock()
        mock_settings_obj.get.side_effect = lambda key, default=None: {
            'config.base_url': 'http://test-api.com',
            'config.model': 'test-model',
            'config.embeddings': 'test-embeddings-model'
        }.get(key, default)
        mock.return_value = mock_settings_obj
        yield mock_settings_obj


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    with patch.dict(os.environ, {
        'FEDOTLLM_LLM_API_KEY': 'test-llm-key',
        'FEDOTLLM_EMBEDDINGS_API_KEY': 'test-embeddings-key',
        'LANGFUSE_PUBLIC_KEY': 'test-langfuse-public',
        'LANGFUSE_SECRET_KEY': 'test-langfuse-secret'
    }):
        yield

@patch('fedotllm.llm.litellm')
def test_query(mock_litellm, mock_env_vars):
    """Test querying with AIInference"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello, world!"))],
    )
    inference = AIInference()
    response = inference.query("Say hello")
    assert response == "Hello, world!"


def test_create_structured_object(mock_env_vars):
    """Test creating a structured object with AIInference"""
    inference = AIInference()
    inference.query = lambda *args, **kwargs: '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}'
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(
        messages="",
        response_model=UserModel
    )
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True
    
def test_create_structured_object_invalid(mock_env_vars):
    """Test creating a structured object that fails validation"""
   
    inference = AIInference()
    inference.query = lambda *args, **kwargs: '{"name": "John Doe", "age": 150, "email": "john@example.com", "active": true}'
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*less than or equal to 120.*") as exc_info:
        user = inference.create(
            messages="",
            response_model=UserModel
        )
        
def test_create_structured_object_missing_field(mock_env_vars):
    """Test creating a structured object with missing fields"""
    inference = AIInference()
    inference.query = lambda *args, **kwargs: '{"name": "John Doe", "age": 30}'
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*[fF]ield required.*") as exc_info:
        user = inference.create(
            messages="",
            response_model=UserModel
        )
        
@pytest.mark.parametrize("response,expected_error", [
    ('name: "John Doe", "age": 30, "email": "john@example.com", "active": true', r".*valid dictionary.*"),
    ('', r".*valid dictionary.*"),
    (None, r".*valid dictionary.*"),   
])
def test_create_structured_object_invalid_format(response, expected_error, mock_env_vars):
    """Test creating a structured object with invalid response"""
    
    inference = AIInference()
    inference.query = lambda *args, **kwargs: response
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=expected_error) as exc_info:
        user = inference.create(
            messages="",
            response_model=UserModel
        )

@pytest.mark.parametrize("response", [
    '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}',
])
@patch('fedotllm.llm.litellm')
def test_create_structured_use_query_valid(mock_litellm, response, mock_env_vars):
    """Test creating a structured object using query method"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))],
    )
    inference = AIInference()
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(
        messages="Create a user object",
        response_model=UserModel
    )
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True
