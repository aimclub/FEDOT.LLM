from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TemplatesConfig(BaseModel):
    code: str
    train: str
    evaluate: str
    predict: str


class AutoMLConfig(BaseModel):
    fix_tries: int = 5
    templates: TemplatesConfig
    predictor_init_kwargs: dict = Field(default_factory=dict)


class LLMConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_headers: Dict[str, Any] = {}
    completion_params: Dict[str, Any] = {}


class EmbeddingsConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_headers: Dict[str, Any] = {}
    embedding_params: Dict[str, Any] = {}


class LangfuseConfig(BaseModel):
    host: str = "https://cloud.langfuse.com"
    public_key: Optional[str] = None
    secret_key: Optional[str] = None


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    automl: AutoMLConfig = Field(default_factory=AutoMLConfig)
