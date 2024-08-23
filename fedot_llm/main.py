from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from fedot_llm.data import Dataset
from fedot_llm.output import BaseFedotAIOutput, ConsoleFedotAIOutput, JupyterFedotAIOutput
from fedot_llm.data.loaders import PathDatasetLoader
from fedot_llm.ai.chains.ready_chains.predict import PredictChain


@dataclass
class FedotAI():
    dataset: Union[Dataset, str]
    model: BaseChatModel
    second_model: Optional[BaseChatModel] = field(default=None, repr=False)
    output: Optional[Union[BaseFedotAIOutput, Literal['jupyter', 'debug']]] = None

    def __post_init__(self):
        if isinstance(self.dataset, str):
            self.dataset = PathDatasetLoader().load(self.dataset)
        

    async def predict(self, dataset_description, visualize=True):
        if isinstance(self.dataset, str):
            raise ValueError("Dataset is not loaded")
        chain = PredictChain(self.model, self.dataset)
        chain_input = {"dataset_description": dataset_description}
        predictions = await self.__start_chain(chain, chain_input)
        return predictions

    async def __start_chain(self, chain: Runnable, chain_input: Dict[str, Any]):
        if self.output:
            if isinstance(self.output, str):
                match self.output:
                    case "jupyter":
                        self.output = JupyterFedotAIOutput()
                    case "debug":
                        self.output = ConsoleFedotAIOutput()
            if isinstance(self.output, BaseFedotAIOutput):
                return await self.output._chain_call(chain=chain, chain_input=chain_input)
            else:
                raise ValueError("Unsupported output type")
