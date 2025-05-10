from fedotllm.llm import AIInference
from pydantic import BaseModel, Field

_llm = AIInference()

class User(BaseModel):
    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., description="Age in years", ge=0, le=120)
    email: str = Field(..., description="Email address")
    role: str = Field(default="user", description="User role in the system")
    score: float = Field(default=0.0, description="User score", ge=0.0, le=100.0)
    is_active: bool = Field(default=True, description="Whether the user account is active")

def test_create_structured_object():
    
    message = """
    New user:
    <name>John Doe</name>
    <age>30</age>
    <email>john@example.com</email>
    <role>admin</role>
    <score>95.5</score>
    <is_active>true</is_active>
    """
    
    user = _llm.create(
        messages=message,
        response_model=User
    )
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.role == "admin"
    assert user.score == 95.5
    assert user.is_active is True
    
def test_create_structured_object_invalid():
    message = """
    New user:
    <name>John Doe</name>
    <age>150</age>
    <email>john@example.com</email>
    <role>admin</role>
    <score>95.5</score>
    <is_active>true</is_active>
    """
    try:
        user = _llm.create(
            messages=message,
            response_model=User
        )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "age" in str(e)
        assert "must be less than or equal to 120" in str(e)