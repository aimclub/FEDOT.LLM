from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from settings.config_loader import get_settings
from pydantic import BaseModel
from typing import Optional
from log import get_logger

logger = get_logger()


class AIInference:

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            self.model = ChatOpenAI(model=model or get_settings().config.model,
                                    base_url=base_url or get_settings().get("config.base_url", None),
                                    api_key=api_key or get_settings()["OPENAI_TOKEN"],
                                    )
        except AttributeError as e:
            raise ValueError("OpenAI key is required") from e

    def chat_completion(
        self,
        user: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        frequency_penalty: float = 0.0,
        *,
        format_vals: dict = {},
        structured: Optional[BaseModel] = None,
        tools: Optional[list] = None,
    ):
        logger.debug(f"Inferring with model: {self.model}".center(100, '='))
        logger.debug(f"User: {user}")
        system and logger.debug(f"System: {system}")
        if structured:
            model = self.model.with_structured_output(structured).bind(
                temperature=temperature, frequency_penalty=frequency_penalty)
        else:
            model = self.model

        if tools:
            model = model.bind_tools(tools)

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
