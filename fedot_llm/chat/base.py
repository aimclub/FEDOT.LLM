import langchain.chat_models as cm
from langchain.chat_models.base import _ConfigurableModel
from typing import Optional, Union, Tuple, List, Literal, Any, overload
from langchain.chat_models.base import BaseChatModel


@overload
def init_chat_model(  # type: ignore[overload-overlap]
        model: str,
        *,
        model_provider: Optional[str] = None,
        configurable_fields: Literal[None] = None,
        config_prefix: Optional[str] = None,
        **kwargs: Any,
) -> BaseChatModel: ...


@overload
def init_chat_model(
        model: Literal[None] = None,
        *,
        model_provider: Optional[str] = None,
        configurable_fields: Literal[None] = None,
        config_prefix: Optional[str] = None,
        **kwargs: Any,
) -> _ConfigurableModel: ...


@overload
def init_chat_model(
        model: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        configurable_fields: Union[Literal["any"], List[str], Tuple[str, ...]] = ...,
        config_prefix: Optional[str] = None,
        **kwargs: Any,
) -> _ConfigurableModel: ...


def init_chat_model(model: Optional[str] = None,
                    *,
                    model_provider: Optional[str] = None,
                    configurable_fields: Optional[
                        Union[Literal["any"], List[str], Tuple[str, ...]]
                    ] = None,
                    config_prefix: Optional[str] = None,
                    **kwargs: Any,
                    ) -> Union[BaseChatModel, _ConfigurableModel]:
    if model_provider == "custom":
        from fedot_llm.chat.custom_web import ChatCustomWeb
        return ChatCustomWeb(model=model, **kwargs)
    else:
        return cm.init_chat_model(
            model=model,
            model_provider=model_provider,
            configurable_fields=configurable_fields,
            config_prefix=config_prefix,
            **kwargs)
