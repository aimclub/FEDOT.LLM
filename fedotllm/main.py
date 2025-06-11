from pathlib import Path
from typing import Callable, List, Optional, Any, AsyncIterator

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.schema import StreamEvent

from fedotllm.agents.supervisor import SupervisorAgent
from fedotllm.agents.agent_wrapper import AgentWrapper
from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.data import Dataset
from fedotllm.llm import AIInference, OpenaiEmbeddings
from fedotllm.agents.translator import TranslatorAgent
from fedotllm.log import logger

CONTENT_PRODUCING_EVENT_NAMES = [
    "SupervisorAgent",
    "AutoMLAgentChat",
    "ResearcherAgent",
]


class TranslatorHandler:
    def __init__(self, translator: TranslatorAgent):
        self.subscribe_events = ["SupervisorAgent", "ResearcherAgent", "AutoMLAgent"]
        self.translated_message_ids = set()
        self.translator = translator

    def handler(self, event: StreamEvent):
        event_name = event.get("name", "")
        data = event.get("data", {})
        if event_name in self.subscribe_events:
            if data:
                output = data.get("output", None)
                if output is not None:
                    if isinstance(output, dict):
                        messages = output.get("messages", None)
                        if messages is not None:
                            if isinstance(messages, list):
                                for message in messages:
                                    if isinstance(message, AIMessage) or isinstance(
                                        message, HumanMessage
                                    ):
                                        if (
                                            message.id
                                            not in self.translated_message_ids
                                        ):
                                            self.translated_message_ids.add(message.id)
                                            message.content = self.translator.translate_output_to_source_language(
                                                message.content
                                            )
                            else:
                                if isinstance(messages, AIMessage) or isinstance(
                                    messages, HumanMessage
                                ):
                                    if messages.id not in self.translated_message_ids:
                                        self.translated_message_ids.add(messages.id)
                                        messages.content = self.translator.translate_output_to_source_language(
                                            messages.content
                                        )


class FedotAI:
    def __init__(
        self,
        task_path: Path | str,
        inference: Optional[AIInference] = None,
        embeddings: Optional[OpenaiEmbeddings] = None,
        handlers: Optional[List[Callable[[StreamEvent], None]]] = None,
        workspace: Path | str | None = None,
    ):
        if isinstance(task_path, str):
            task_path = Path(task_path)
        self.task_path = task_path.resolve()
        assert self.task_path.is_dir(), (
            "Task path does not exist or is not a directory."
        )

        self.inference = inference if inference is not None else AIInference()
        self.embeddings = embeddings if embeddings is not None else OpenaiEmbeddings()
        self.handlers = handlers if handlers is not None else []

        if isinstance(workspace, str):
            workspace = Path(workspace)
        self.workspace = workspace

    async def ainvoke(self, message: str):
        logger.info(
            f"FedotAI ainvoke called. Input message (first 100 chars): '{message[:100]}...'"
        )
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Workspace for ainvoke created at: {self.workspace}")

        dataset = Dataset.from_path(self.task_path)
        translator_agent = TranslatorAgent(inference=self.inference)

        logger.info("FedotAI ainvoke: Translating input message to English.")
        translated_message = translator_agent.translate_input_to_english(message)
        logger.info(
            f"FedotAI ainvoke: Input message translated to (first 100 chars): '{translated_message[:100]}...'"
        )

        automl_agent = AutoMLAgentChat(
            inference=self.inference, dataset=dataset, workspace=self.workspace
        ).create_graph()

        researcher_agent = AgentWrapper(
            ResearcherAgent(inference=self.inference, embeddings=self.embeddings)
        ).create_graph()

        entry_point = SupervisorAgent(
            inference=self.inference,
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
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Workspace created at: {self.workspace}")
        dataset = Dataset.from_path(self.task_path)
        translator_agent = TranslatorAgent(inference=self.inference)
        logger.info("Translating input message to English for ask.")
        translated_message = translator_agent.translate_input_to_english(message)
        logger.info(
            f"Input message translated to (first 100 chars): '{translated_message[:100]}...'"
        )

        automl_agent = AutoMLAgentChat(
            inference=self.inference, dataset=dataset, workspace=self.workspace
        ).create_graph()
        researcher_agent = AgentWrapper(
            ResearcherAgent(inference=self.inference, embeddings=self.embeddings)
        ).create_graph()
        entry_point = SupervisorAgent(
            inference=self.inference,
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
