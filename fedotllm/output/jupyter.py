from typing import List

from IPython.display import display, Markdown, DisplayObject, clear_output
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel, Field, ConfigDict


class JupyterOutput(BaseModel):
    display_content: List[DisplayObject] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def messages_handler(self):
        subscribe_events = ["SupervisorAgent", "ResearcherAgent", "AutoMLAgent"]
        content = []
        message_idx = set()

        def handler(event: StreamEvent):
            nonlocal subscribe_events, content
            event_name = event.get("name", "")
            header_line_len = 50
            data = event.get("data", {})
            if event_name in subscribe_events:
                if data:
                    output = data.get("output", None)
                    if output is not None:
                        if isinstance(output, dict):
                            messages = output.get("messages", None)
                            if messages is not None:
                                new_messages = []
                                if isinstance(messages, list):
                                    for message in messages:
                                        if isinstance(message, AIMessage) or isinstance(
                                            message, HumanMessage
                                        ):
                                            if message.id not in message_idx:
                                                message_idx.add(message.id)
                                                new_messages.append(message)
                                else:
                                    if isinstance(messages, AIMessage) or isinstance(
                                        messages, HumanMessage
                                    ):
                                        if messages.id not in message_idx:
                                            message_idx.add(messages.id)
                                            new_messages.append(messages)
                                for message in new_messages:
                                    if isinstance(message, AIMessage):
                                        content.append(
                                            " Supervisor ".center(header_line_len, "=")
                                        )
                                        content.append(message.content)
                                    if isinstance(message, HumanMessage):
                                        if message.name:
                                            content.append(
                                                f" {message.name} ".center(
                                                    header_line_len, "="
                                                )
                                            )
                                        else:
                                            content.append(
                                                " HumanMessage ".center(
                                                    header_line_len, "="
                                                )
                                            )
                                        content.append(message.content)
                            if len(content) > 0:
                                self.display_content.append(
                                    Markdown("\n\n".join(content))
                                )

        return handler

    def display_handler(self):
        def handler(_: StreamEvent):
            if self.display_content:
                clear_output(wait=True)
                for content in self.display_content:
                    display(content)
                self.display_content = []

        return handler

    @property
    def subscribe(self):
        return [self.messages_handler(), self.display_handler()]
