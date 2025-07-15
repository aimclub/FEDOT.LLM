from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError
from tenacity import wait_none

from fedotllm.configs.schema import LLMConfig
from fedotllm.llm import AIInference


class UserModel(BaseModel):
    """Test model for structured response testing"""

    name: str = Field(..., description="User name")
    age: int = Field(..., description="User age", ge=0, le=120)
    email: str = Field(..., description="User email")
    active: bool = Field(default=True, description="User active status")


# Fixtures
@pytest.fixture
def llm_config():
    config = LLMConfig(
        provider="test-provider",
        model_name="test-model",
        base_url="https://test.api.com",
        api_key="sk-12345",
        extra_headers={"X-Title": "FEDOT.LLM-Test"},
        completion_params={"temperature": 1.0},
    )
    return config


@patch("fedotllm.llm.litellm")
def test_query(mock_litellm, llm_config):
    """Test querying with AIInference"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello, world!"))],
    )
    inference = AIInference(llm_config)
    response = inference.query("Say hello")
    assert response == "Hello, world!"


def test_create_structured_object(llm_config):
    """Test creating a structured object with AIInference"""
    inference = AIInference(llm_config)
    inference.query = (
        lambda *args,
        **kwargs: '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}'
    )
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(messages="", response_model=UserModel)
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True


def test_create_structured_object_invalid(llm_config):
    """Test creating a structured object that fails validation"""

    inference = AIInference(llm_config)
    inference.query = (
        lambda *args,
        **kwargs: '{"name": "John Doe", "age": 150, "email": "john@example.com", "active": true}'
    )
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*less than or equal to 120.*"):
        inference.create(messages="", response_model=UserModel)


def test_create_structured_object_missing_field(llm_config):
    """Test creating a structured object with missing fields"""
    inference = AIInference(llm_config)
    inference.query = lambda *args, **kwargs: '{"name": "John Doe", "age": 30}'
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*[fF]ield required.*"):
        inference.create(messages="", response_model=UserModel)


@pytest.mark.parametrize(
    "response,expected_error",
    [
        (
            'name: "John Doe", "age": 30, "email": "john@example.com", "active": true',
            r".*valid dictionary.*",
        ),
        ("", r".*valid dictionary.*"),
        (None, r".*valid dictionary.*"),
    ],
)
def test_create_structured_object_invalid_format(response, expected_error, llm_config):
    """Test creating a structured object with invalid response"""

    inference = AIInference(llm_config)
    inference.query = lambda *args, **kwargs: response
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=expected_error):
        inference.create(messages="", response_model=UserModel)


@pytest.mark.parametrize(
    "response",
    [
        '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}',
    ],
)
@patch("fedotllm.llm.litellm")
def test_create_structured_use_query_valid(mock_litellm, response, llm_config):
    """Test creating a structured object using query method"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))],
    )
    inference = AIInference(llm_config)
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(messages="Create a user object", response_model=UserModel)
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True
