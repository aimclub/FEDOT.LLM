import json
import os

from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import Any, List, Dict, Literal, Optional, Union
from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from fedot_llm.chains import ChainBuilder
from fedot_llm.data import Dataset
from fedot_llm.output import BaseFedotAIOutput, ConsoleFedotAIOutput, JupyterFedotAIOutput
from fedot_llm.data.loaders import PathDatasetLoader

class BenchmarkResult():
    def __init__(self, predictions, targets, predictions_raw = None):
        self.predictions_raw = predictions if predictions_raw == None else predictions_raw
        self.predictions = predictions
        self.targets = targets
        self.metrics = {}

@dataclass
class BaseBenchmarkRun(ABC):
    dataset: List[Union[Dataset, Path, str]]
    model: InitVar[BaseChatModel]
    second_model: InitVar[Optional[BaseChatModel]] = field(default=None)
    ignore_files: InitVar[List[Union[Path, str]]] = field(default=None)
    chain_builder: ChainBuilder = field(init=False)
    output: Optional[Union[BaseFedotAIOutput, Literal['jupyter', 'debug']]] = None

    def __post_init__(self, model, second_model, ignore_files):
        if isinstance(self.dataset, (Path, str)):
            self.dataset = PathDatasetLoader().load(self.dataset, ignore_files = ignore_files)
        if second_model is None:
            second_model = model
        self.chain_builder = ChainBuilder(
            assistant=model, dataset=self.dataset, arbiter=second_model)

    async def _start_chain(self, chain: Runnable, chain_input: Dict[str, Any], 
                            finish_event_name):
        if self.output:
            if isinstance(self.output, str):
                match self.output:
                    case "jupyter":
                        self.output = JupyterFedotAIOutput(
                            finish_event_name = finish_event_name)
                    case "debug":
                        self.output = ConsoleFedotAIOutput(
                            finish_event_name = finish_event_name)
            if isinstance(self.output, BaseFedotAIOutput):
                return await self.output._chain_call(chain=chain, chain_input=chain_input)
            else:
                raise ValueError("Unsupported output type")

    @abstractmethod
    async def predict(self, dataset_description: Dict[str, Any]) -> BenchmarkResult:
        """ 
        Perform the benchmark evaluation run 
        """

@dataclass
class BaseBenchmark(ABC):
    model: BaseChatModel
    second_model: Optional[BaseChatModel] = None
    datasets_metadata_path: Union[Path, str] = field(default = None)
    datasets_store_path: Union[Path, str] = field(default = Path('../../datasets-local'))
    extra_data_file_name: Union[Path, str] = field(default = 'extra.csv')
    dataset_metadatas: List = None
    output: Optional[Union[BaseFedotAIOutput, Literal['jupyter', 'debug']]] = None

    def __post_init__(self):
        if not os.path.exists(self.datasets_store_path):
            os.makedirs(self.datasets_store_path)
        if os.path.exists(self.datasets_metadata_path):
            with open(self.datasets_metadata_path) as f:
                self.dataset_metadatas = json.load(f)
        if self.second_model is None:
            self.second_model = self.model

    @abstractmethod
    async def predict(self) -> BenchmarkResult:
        """ 
        Initialize loading of dataset and evaluation of benchmark
        """

    @abstractmethod
    def display_results(self, result: BenchmarkResult):
        """ 
        Display results in a task-specific way

        Args:
            result: acquired benchmark run results
        """