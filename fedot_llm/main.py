from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Literal, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from fedot_llm.chains import ChainBuilder, FedotPredictions
from fedot_llm.data import Dataset
from fedot_llm.output import BaseFedotAIOutput, ConsoleFedotAIOutput, JupyterFedotAIOutput
from fedot_llm.data.loaders import PathDatasetLoader


@dataclass
class FedotAI():
    dataset: Union[Dataset, str]
    model: InitVar[BaseChatModel]
    second_model: InitVar[Optional[BaseChatModel]] = field(default=None, repr=False)
    chain_builder: ChainBuilder = field(init=False)
    output: Optional[Union[BaseFedotAIOutput, Literal['jupyter', 'debug']]] = None

    def __post_init__(self, model, second_model):
        if isinstance(self.dataset, str):
            self.dataset = PathDatasetLoader().load(self.dataset)
        if second_model is None:
            second_model = model
        self.chain_builder = ChainBuilder(
            assistant=model, dataset=self.dataset, arbiter=second_model)

    async def predict(self, dataset_description, visualize=True):
        chain = self.chain_builder.predict_chain
        chain_input = {"big_description": dataset_description}
        predictions = await self.__start_chain(chain, chain_input)
        predictions: Optional[FedotPredictions] = predictions
        if visualize and predictions:
            predictions.best_pipeline.show()
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
