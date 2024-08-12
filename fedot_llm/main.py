from dataclasses import dataclass, field, InitVar
from typing import Optional, Literal, Any
from fedot_llm.data import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from fedot_llm.chains import ChainBuilder, FedotPredictions
from IPython.display import display, clear_output, Markdown
from fedot_llm.chains import stages
from langchain_core.runnables import Runnable
import logging 
import asyncio

@dataclass
class FedotAI():
    dataset_dir: str
    description: str
    model: InitVar[BaseChatModel]
    second_model: InitVar[Optional[BaseChatModel]] = field(default=None)
    display: Optional[Literal['jupyter', 'debug']] = None
    dataset: Dataset = field(init=False)
    chain_builder: ChainBuilder = field(init=False)
   
    def __post_init__(self, model, second_model):
        self.dataset = Dataset.load_from_path(self.dataset_dir)
        if second_model is None:
            second_model = model
        self.chain_builder = ChainBuilder(assistant=model, dataset=self.dataset, arbiter=second_model)

    async def predict(self, visualize=True):
        chain = self.chain_builder.predict_chain
        logger_task = self.__logger_init(chain)
        predictions_task = self.__chain_call(chain, visualize)
        logger_result, predictions = await asyncio.gather(logger_task, predictions_task)
        predictions: Optional[FedotPredictions] = predictions
        if visualize and predictions:
            predictions.best_pipeline.show()
        return predictions
        
    
    async def __chain_call_debug(self, chain: Runnable):
        async for event in chain.astream_events({'big_description': self.description}, version="v2"):
            print(event)
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return event['data']['output']
            
    async def __logger_init(self, chain: Runnable):
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(filename='llm.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        async for event in chain.astream_events({'big_description': self.description}, version="v2"):
            log_msg = []
            if event['event'] == 'on_chat_model_end':
                if event.get('data', None) and 'chunk' not in event['data']:
                    if event['data'].get('input', None):
                        log_msg.append(f"INPUT:\n{event['data']['input']['messages']}")
                    if event['data'].get('output', None):
                        log_msg.append(f"ANSWER:\n{event['data']['output'].content}")
                    logger.debug(f"{event['name']}\n" + '\n'.join(log_msg))
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return
        
    
    async def __chain_call_jupyter(self, chain: Runnable):
        include_names=[str(stage) for stage in stages]
        include_names.append('master')
        messages = '\n'
        async for event in chain.astream_events({'big_description': self.description}, version="v2", include_names=include_names, include_tags=['print']):
            clear_output(wait=True)
            display_str = '# Progress:\n'
            for stage in stages:
                if stage.name in event['name']:
                    if event['event'] == 'on_chain_start':
                        stage.status = 'Running'
                    elif event['event'] == 'on_chain_stream':
                        stage.status = 'Streaming'
                    elif event['event'] == 'on_chain_end':
                        stage.status = 'Сompleted'
                    
            for stage in stages:
                if stage.status == 'Waiting':
                    display_str += f"- [] {stage.display_name}\n"
                if stage.status == 'Running' or stage.status == 'Streaming':
                    display_str += f"- () {stage.display_name}\n"
                elif stage.status == 'Сompleted':
                    display_str += f"- [x] {stage.display_name}\n"
            
            if 'print' in event['tags']:
                messages += event['data'].get('chunk', '')
                    
            display(Markdown(display_str + messages))
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return event['data']['output']
    
    async def __chain_call(self, chain: Runnable, visualize: bool):
        if self.display is None:
            return await chain.ainvoke({'big_description': self.description})
        elif self.display == 'jupyter':
            return await self.__chain_call_jupyter(chain)
        elif self.display == 'debug':
            return await self.__chain_call_debug(chain)
        else: 
            raise ValueError(f'Unsupported display type: {self.display}')