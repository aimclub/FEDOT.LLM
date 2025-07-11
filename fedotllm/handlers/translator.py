from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent

from fedotllm.agents.translator import TranslatorAgent


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
