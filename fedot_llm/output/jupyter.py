from dataclasses import dataclass, field
from typing import List, Dict

from IPython.display import display, Markdown, DisplayObject, clear_output
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent

from fedot_llm.ai.actions import Actions, Action
from fedot_llm.ai.chains.analyze import AnalyzeFedotResultChain
from fedot_llm.ai.chains.fedot import FedotPredictChain
from fedot_llm.ai.chains.metainfo import DefineDatasetChain, DefineSplitsChain, DefineTaskChain


@dataclass
class JupyterOutput:
    display_content: List[DisplayObject] = field(default_factory=list)

    def automl_progress_handler(self):
        actions: Actions = Actions([
            Action.from_chain(DefineDatasetChain),
            Action.from_chain(DefineSplitsChain),
            Action.from_chain(DefineTaskChain),
            Action.from_chain(FedotPredictChain),
            Action.from_chain(AnalyzeFedotResultChain)
        ])
        records: Dict[Action, str] = {}

        def on_change_hook(event: StreamEvent, action: Action) -> None:
            nonlocal records
            content = ''
            if action.state == 'Waiting':
                content += f"- [] {action}"
            elif action.state == 'Running' or action.state == 'Streaming':
                content += f"- () {action}"
            elif action.state == 'Completed':
                content += f"- [X] {action}"
            records[action] = content
            self.display_content.append(Markdown('\n\n'.join(records.values())))

        for action in actions.records.values():
            action.on_change.append(on_change_hook)
            records[action] = f"- [] {action}\n\n"

        return actions.handler

    def messages_handler(self):
        subscribe_events = ['SupervisorAgent', 'ResearcherAgent', 'AutoMLAgent']
        content = []
        message_idx = set()

        def handler(event: StreamEvent):
            nonlocal subscribe_events, content
            event_name = event.get('name', '')
            header_line_len = 50
            data = event.get('data', {})
            if event_name in subscribe_events:
                if data:
                    output = data.get('output', None)
                    if output is not None:
                        if isinstance(output, dict):
                            messages = output.get('messages', None)
                            if messages is not None:
                                new_messages = []
                                if isinstance(messages, list):
                                    for message in messages:
                                        if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
                                            if message.id not in message_idx:
                                                message_idx.add(message.id)
                                                new_messages.append(message)
                                else:
                                    if isinstance(messages, AIMessage) or isinstance(messages, HumanMessage):
                                        if messages.id not in message_idx:
                                            message_idx.add(messages.id)
                                            new_messages.append(messages)
                                for message in new_messages:
                                    if isinstance(message, AIMessage):
                                        content.append(" Supervisor ".center(header_line_len, '='))
                                        content.append(message.content)
                                    if isinstance(message, HumanMessage):
                                        if message.name:
                                            content.append(f" {message.name} ".center(header_line_len, '='))
                                        else:
                                            content.append(" HumanMessage ".center(header_line_len, '='))
                                        content.append(message.content)
                            if len(content) > 0:
                                self.display_content.append(Markdown('\n\n'.join(content)))

        return handler

    def display_handler(self):
        def handler(event: StreamEvent):
            if self.display_content:
                clear_output(wait=True)
                for content in self.display_content:
                    display(content)
                self.display_content = []

        return handler

    @property
    def subscribe(self):
        return [self.automl_progress_handler(), self.messages_handler(), self.display_handler()]
