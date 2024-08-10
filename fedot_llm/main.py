from dataclasses import dataclass, field, InitVar
from typing import Optional, Literal
from fedot_llm.data import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from fedot_llm.chains import ChainBuilder
from IPython.display import display, clear_output, Markdown
from fedot_llm.chains import stages
from langchain_core.runnables import Runnable

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

    async def predict(self):
        return await self.__chain_call(self.chain_builder.predict_chain)
    
    async def __chain_call_debug(self, chain: Runnable):
        async for event in chain.astream_events({'big_description': self.description}, version="v2"):
            print(event)
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return event['data']['output']
    
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
    
    async def __chain_call(self, chain: Runnable):
        if self.display is None:
            return await chain.ainvoke({'big_description': self.description})
        elif self.display == 'jupyter':
            return await self.__chain_call_jupyter(chain)
        elif self.display == 'debug':
            return await self.__chain_call_debug(chain)
        else: 
            raise ValueError(f'Unsupported display type: {self.display}')