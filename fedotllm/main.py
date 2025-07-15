from pathlib import Path
from typing import Any, AsyncIterator, Callable, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent

from fedotllm.agents.automl import AutoMLAgent
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.supervisor import SupervisorAgent
from fedotllm.agents.translator import TranslatorAgent
from fedotllm.configs.loader import load_config
from fedotllm.handlers.translator import TranslatorHandler
from fedotllm.llm import AIInference, LiteLLMEmbeddings
from fedotllm.log import logger


class FedotAI:
    def __init__(
        self,
        task_path: str | Path,
        handlers: Optional[List[Callable[[StreamEvent], None]]] = None,
        workspace: Optional[str | Path] = None,
        presets: Optional[str | Path | List[str | Path]] = None,
        config_path: Optional[str | Path] = None,
        config_overrides: Optional[List[str]] = None,
    ):
        """
        Initialize the FedotAI agent system.

        Args:
            task_path (str): Path to the dataset or task directory.
            handlers (Optional[List[Callable[[StreamEvent], None]]]): Optional list of event handler callbacks for streaming events.
            workspace (Optional[str]): Optional path to the workspace directory for outputs and artifacts.
            presets (Optional[str | List[str]]): Optional preset or list of presets for configuration.
            config_path (Optional[str]): Optional path to a custom configuration file.
            config_overrides (Optional[List[str]]): Optional list of configuration override strings.

        Raises:
            AssertionError: If the provided task_path does not exist or is not a directory.
        """
        self.task_path = Path(task_path).resolve()
        assert self.task_path.is_dir(), (
            "Task path does not exist or is not a directory."
        )

        self.config = load_config(
            presets=presets, config_path=config_path, overrides=config_overrides
        )

        self.inference = AIInference(self.config.llm)
        self.embeddings = LiteLLMEmbeddings(self.config.embeddings)
        self.handlers = handlers if handlers is not None else []
        self.config.session_id = self.config.session_id or pd.Timestamp.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.workspace = workspace

    async def ainvoke(self, message: str):
        logger.info(
            f"FedotAI ainvoke called. Input message (first 100 chars): '{message[:100]}...'"
        )
        if not self.workspace:
            self.workspace = Path(f"fedotllm-output-{self.config.session_id}")
            logger.info(f"Workspace for ainvoke created at: {self.workspace}")

        translator_agent = TranslatorAgent(inference=self.inference)

        logger.info("FedotAI ainvoke: Translating input message to English.")
        translated_message = translator_agent.translate_input_to_english(message)
        logger.info(
            f"FedotAI ainvoke: Input message translated to (first 100 chars): '{translated_message[:100]}...'"
        )

        automl_agent = AutoMLAgent(
            config=self.config, dataset_path=self.task_path, workspace=self.workspace
        ).create_graph()
        researcher_agent = ResearcherAgent(config=self.config).create_graph()

        entry_point = SupervisorAgent(
            config=self.config,
            automl_agent=automl_agent,
            researcher_agent=researcher_agent,
        ).create_graph()

        raw_response = await entry_point.ainvoke(
            {"messages": [HumanMessage(content=translated_message)]}
        )
        logger.debug(
            f"FedotAI ainvoke: Raw response from SupervisorAgent: {raw_response}"
        )

        if (
            raw_response
            and "messages" in raw_response
            and isinstance(raw_response["messages"], list)
            and len(raw_response["messages"]) > 0
        ):
            last_message_original = raw_response["messages"][-1]
            logger.debug(
                f"FedotAI ainvoke: Original last_message from Supervisor: {last_message_original}"
            )

            if hasattr(last_message_original, "content"):
                ai_message_content = last_message_original.content
                logger.info(
                    f"FedotAI ainvoke: Before output translation. Source lang: {translator_agent.source_language}. Content (first 100): '{ai_message_content[:100]}...'"
                )

                translated_output = (
                    translator_agent.translate_output_to_source_language(
                        ai_message_content
                    )
                )
                logger.info(
                    f"FedotAI ainvoke: After output translation. Translated content (first 100): '{translated_output[:100]}...'"
                )

                if isinstance(last_message_original, AIMessage):
                    # Create new AIMessage, preserving other attributes
                    # Ensure all attributes are correctly handled, using defaults if necessary
                    new_ai_message = AIMessage(
                        content=translated_output,
                        id=getattr(last_message_original, "id", None),
                        response_metadata=getattr(
                            last_message_original, "response_metadata", {}
                        ),
                        tool_calls=getattr(last_message_original, "tool_calls", []),
                        tool_call_chunks=getattr(
                            last_message_original, "tool_call_chunks", []
                        ),
                        usage_metadata=getattr(
                            last_message_original, "usage_metadata", None
                        ),
                    )
                    raw_response["messages"][-1] = new_ai_message
                    logger.debug(
                        f"FedotAI ainvoke: Updated AIMessage with translated content: {new_ai_message}"
                    )
                else:
                    logger.warning(
                        f"FedotAI ainvoke: Last message is not AIMessage (type: {type(last_message_original)}), direct content update might be insufficient or ineffective if immutable."
                    )
                    # Attempting to update content directly if mutable, though AIMessage is preferred.
                    if hasattr(last_message_original, "content"):
                        last_message_original.content = translated_output
                        logger.debug(
                            f"FedotAI ainvoke: Attempted to update content of non-AIMessage. New last_message: {last_message_original}"
                        )

            else:
                logger.warning(
                    "FedotAI ainvoke: Last message in response has no 'content' attribute."
                )
        else:
            logger.warning(
                "FedotAI ainvoke: No messages found in raw_response or response structure is unexpected."
            )

        return raw_response

    async def ask(self, message: str) -> AsyncIterator[Any]:
        logger.info(
            f"FedotAI ask called with message (first 100 chars): '{message[:100]}...'"
        )
        if not self.workspace:
            self.workspace = Path(f"fedotllm-output-{self.config.session_id}")
            logger.info(f"Workspace created at: {self.workspace}")
        translator_agent = TranslatorAgent(inference=self.inference)
        logger.info("Translating input message to English for ask.")
        translated_message = translator_agent.translate_input_to_english(message)
        logger.info(
            f"Input message translated to (first 100 chars): '{translated_message[:100]}...'"
        )

        automl_agent = AutoMLAgent(
            config=self.config, dataset_path=self.task_path, workspace=self.workspace
        ).create_graph()
        researcher_agent = ResearcherAgent(config=self.config).create_graph()
        entry_point = SupervisorAgent(
            config=self.config,
            automl_agent=automl_agent,
            researcher_agent=researcher_agent,
        ).create_graph()

        translator_handler = TranslatorHandler(translator_agent)
        async for event in entry_point.astream_events(
            {"messages": [HumanMessage(content=translated_message)]}, version="v2"
        ):
            translator_handler.handler(event)

            for handler in self.handlers:
                handler(event)
            yield event
