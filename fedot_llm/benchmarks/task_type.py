import os

from pathlib import Path
from dataclasses import dataclass, field
from sklearn.metrics import f1_score
from typing import Union

from fedot_llm.data.fetchers.benchmark_fetchers import fetch_and_save_all_datasets

import pandas as pd

from fedot_llm.benchmarks.base import BenchmarkResult, BaseBenchmarkRun, BaseBenchmark

@dataclass
class TaskTypeBenchmarkRun(BaseBenchmarkRun):

    async def predict(self, dataset_description):
        chain = self.chain_builder.dataset_task_type_chain
        chain_input = {"detailed_description": dataset_description}
        predictions = await self._start_chain(chain, chain_input, 
                                                finish_event_name = "dataset_task_type_chain")
        return predictions   


@dataclass
class TaskTypeBenchmark(BaseBenchmark):
    datasets_metadata_path: Union[Path, str] = field(
        default=Path('../../datasets/dataset_descriptions/task_type_descriptions.json'))

    async def predict(self):
        fetch_and_save_all_datasets(self.dataset_metadatas, self.datasets_store_path)

        predictions = []
        targets = []

        for metadata in self.dataset_metadatas:
            dataset_path = self.datasets_store_path / metadata['name']
            description = metadata['description']
            benchmark_run = TaskTypeBenchmarkRun(
                dataset = dataset_path,
                ignore_files = [self.extra_data_file_name], 
                model = self.model,
                second_model = self.second_model,
                output = self.output)
            prediction = await benchmark_run.predict(description)
            predictions.append(prediction)
            targets.append(metadata['type'])
        
        results = BenchmarkResult(predictions, targets) 
        results.metrics["f1"] = f1_score(predictions, targets, pos_label="regression")
        return results
    
    def display_results(self, result: BenchmarkResult):
        for metric, value in result.metrics.items():
            print("{}: {}".format(metric, value))
        print("\n")

        for i, metadata in enumerate(self.dataset_metadatas):
            print("Dataset: {}".format(metadata['name']))
            print("Task Description: {}".format(metadata['description']))
            print("Target type: {}".format(result.targets[i]))
            print("Predicted type: {}".format(result.predictions[i]))
            print("\n")
        
