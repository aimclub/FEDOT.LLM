import logging
from typing import Any, Dict

from IPython.display import Markdown, clear_output, display
from langchain_core.runnables import Runnable

from fedot_llm.chains import steps
from fedot_llm.output.base import BaseFedotAIOutput


class JupyterFedotAIOutput(BaseFedotAIOutput):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(
            filename='llm.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

    async def _chain_call(self, chain: Runnable, chain_input: Dict[str, Any]):
        messages = '\n'
        async for event in chain.astream_events(chain_input, version="v2"):
            clear_output(wait=True)
            log_msg = []
            if event['event'] == 'on_chat_model_end':
                if event.get('data', None) and 'chunk' not in event['data']:
                    if event['data'].get('input', None):
                        log_msg.append(
                            f"INPUT:\n{event['data']['input']['messages']}")
                    if event['data'].get('output', None):
                        log_msg.append(
                            f"ANSWER:\n{event['data']['output'].content}")
                    self.logger.debug(
                        f"{event['name']}\n" + '\n'.join(log_msg))
            display_str = '# Progress:\n'
            for step in steps:
                if step.id in event['name']:
                    if event['event'] == 'on_chain_start':
                        step.status = 'Running'
                    elif event['event'] == 'on_chain_stream':
                        step.status = 'Streaming'
                    elif event['event'] == 'on_chain_end':
                        step.status = 'Сompleted'

            for step in steps:
                if step.status == 'Waiting':
                    display_str += f"🏃‍♂️‍➡️ {step}"
                if step.status == 'Running' or step.status == 'Streaming':
                    display_str += f"- () {step.name}\n"
                elif step.status == 'Сompleted':
                    display_str += f"✅ {step.name}\n"

            if 'print' in event['tags']:
                messages += event['data'].get('chunk', '')

            display(Markdown(display_str + messages))
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return event['data']['output']
