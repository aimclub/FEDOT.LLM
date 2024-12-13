from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


class AIInference:

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            headers = None
            if "vsegpt" in base_url:
                headers = {"X-Title": "FEDOT.LLM"}
            self.model: ChatOpenAI = ChatOpenAI(model=model or get_settings().config.model,
                                                base_url=base_url or get_settings().get("config.base_url", None),
                                                api_key=SecretStr(
                                                    api_key or get_settings()["OPENAI_TOKEN"]),
                                                default_headers=headers,
                                                )
        except AttributeError as e:
            raise ValueError("OpenAI key is required") from e

    def chat_completion(
            self,
            user: str,
            system: Optional[str] = None,
            temperature: float = 0.2,
            frequency_penalty: float = 0.0,
            max_tokens: int = 3000,
            *,
            format_vals=None,
            structured: Optional[BaseModel] = None,
            tools: Optional[list] = None,
    ):
        if format_vals is None:
            format_vals = {}
        logger.debug(f"Inferring with model: {self.model}".center(100, '='))
        logger.debug(f"User: {user}")
        system and logger.debug(f"System: {system}")
        model = self.model.bind(
            temperature=temperature, 
            frequency_penalty=frequency_penalty, 
            max_completion_tokens=max_tokens)
        if tools:
            model = model.bind_tools(tools)
        if structured:
            model = model.with_structured_output(structured)

        template: ChatPromptTemplate | PromptTemplate
        if system:
            template = ChatPromptTemplate(
                [('system', system), ('user', user)], template_format='jinja2')
        else:
            template = PromptTemplate.from_template(
                user, template_format='jinja2')

        execution_chain = template | model

        result = execution_chain.invoke(format_vals)
        logger.debug(f"Result: {result.__str__()}")
        logger.debug("=" * 100)
        return result
