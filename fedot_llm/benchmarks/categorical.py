import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field

from sklearn.metrics import f1_score, recall_score
from tabulate import tabulate

from fedot_llm.data.fetchers.benchmark_fetchers import fetch_and_save_all_datasets
from fedot_llm.benchmarks.base import BenchmarkResult, BaseBenchmarkRun, BaseBenchmark

@dataclass
class CategoricalBenchmarkRun(BaseBenchmarkRun):

    async def predict(self, dataset_description):
        chain = self.chain_builder.master_chain
        chain_input = {"big_description": dataset_description}
        predictions = await self._start_chain(chain, chain_input, 
                                                finish_event_name = "categorize_runnable")
        return predictions   

@dataclass
class CategoricalBenchmark(BaseBenchmark):
    datasets_metadata_path: str = field(
        default=Path('../../datasets/dataset_descriptions/categorical_benchmark/categorical_descriptions_short.json'))

    @staticmethod
    def _get_benchmark_result(predictions, predictions_raw, targets):
        result = BenchmarkResult(predictions = predictions, 
                            predictions_raw = predictions_raw,
                            targets = targets) 

        predictions_cat = [[y == "categorical" for y in x] for x in predictions]
        targets_cat = [[y == "categorical" for y in x] for x in targets]
        f1_scores = [f1_score(pred, targ, pos_label= True) for pred, targ in zip(predictions_cat, targets_cat)]
        recall_scores = [recall_score(pred, targ, pos_label= True) for pred, targ in zip(predictions_cat, targets_cat)]

        result.metrics = {
            "f1_mean": np.mean(f1_scores),
            "recall_mean": np.mean(recall_scores)
        }
        return result

    async def predict(self):
        fetch_and_save_all_datasets(self.dataset_metadatas, self.datasets_store_path)

        predictions = []
        predictions_raw = []
        targets = []

        for metadata in self.dataset_metadatas:
            dataset_path = self.datasets_store_path / metadata['name']
            extra_data_path = dataset_path / self.extra_data_file_name
            extra_data = pd.read_csv(extra_data_path)
            description = metadata['description']
            benchmark_run = CategoricalBenchmarkRun(
                dataset = dataset_path,
                ignore_files = [self.extra_data_file_name], 
                model = self.model,
                second_model = self.second_model,
                output = self.output)
            prediction = await benchmark_run.predict(description)
            predictions_raw.append(prediction)

            #Index is also categorized, therefore we skip the first - may be removed in the future
            prediction = [x['category'].column_type.lower() for x in prediction[1:]]
            target = [x.lower() for x in extra_data.type]
            predictions.append(prediction)
            targets.append(target)

        return self._get_benchmark_result(predictions = predictions, 
                                          predictions_raw = predictions_raw,
                                          targets = targets) 
    
    def display_results(self, result: BenchmarkResult):
        for metric, value in result.metrics.items():
            print("{}: {}".format(metric, value))
        print("\n")

        for i, metadata in enumerate(self.dataset_metadatas):
            
            print("Dataset: {}".format(metadata['name']))
            print("Task Description: {}".format(metadata['description']))

            predictions_raw = result.predictions_raw[i][1:]
            columns = [x['category'].column_type.name for x in predictions_raw]

            df = pd.DataFrame.from_dict({
                    'column': columns,
                    'predicted': result.predictions[i],
                    'target': result.targets[i], 
            })
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            print("\n")