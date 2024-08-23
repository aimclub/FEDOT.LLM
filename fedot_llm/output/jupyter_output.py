import logging
from typing import Any, Dict

from IPython.display import Markdown, clear_output, display
from langchain_core.runnables import Runnable
from fedot_llm.output.base import BaseFedotAIOutput
from fedot_llm.ai.stages import Stages, Stage
from fedot_llm.ai.chains.metainfo import DefineDatasetChain, DefineSplitsChain, DefineTaskChain
from fedot_llm.ai.chains.fedot import FedotPredictChain
from fedot_llm.ai.chains.analyze import AnalyzeFedotResultChain

class JupyterFedotAIOutput(BaseFedotAIOutput):
    stages: Stages = Stages([
        Stage.from_chain(DefineDatasetChain),
        Stage.from_chain(DefineSplitsChain),
        Stage.from_chain(DefineTaskChain),
        Stage.from_chain(FedotPredictChain),
        Stage.from_chain(AnalyzeFedotResultChain)
    ])
    
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
            
            
            if event['name'] in self.stages().keys():
                if event['event'] == 'on_chain_start':
                    self.stages()[event['name']].state = 'Running'
                elif event['event'] == 'on_chain_stream':
                    self.stages.data[event['name']].state = 'Streaming'
                elif event['event'] == 'on_chain_end':
                    self.stages.data[event['name']].state = 'Completed'
    
            for item in self.stages.data.values():
                if item.state == 'Waiting':
                    display_str += f"- [] {item.name}\n"
                elif item.state == 'Running' or item.state == 'Streaming':
                    display_str += f"- () {item.name}\n"
                elif item.state == 'Completed':
                    display_str += f"- [x] {item.name}\n"

            if 'print' in event['tags']:
                messages += event['data'].get('chunk', '')

            display(Markdown(display_str + messages))
            if event['name'] == chain.__class__.__name__ and event['event'] == 'on_chain_end':
                return event['data']['output']
