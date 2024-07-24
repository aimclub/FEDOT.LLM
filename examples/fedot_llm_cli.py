import argparse
import logging
import os
import re
from pprint import pprint
from sklearn.metrics import accuracy_score

from fedot_llm.data.data import Dataset
from fedot_llm.fedot_util import run_example
from fedot_llm.language_models import prompts
from fedot_llm.language_models.actions import ModelAction
from fedot_llm.language_models.llms import HuggingFaceLLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main(dataset_name, dataset_description):
    dataset = load_data(dataset_name)

    model = HuggingFaceLLM(
        model_id="microsoft/Phi-3-mini-4k-instruct", max_new_tokens=500
    )
    action = ModelAction(model=model)

    data_clarification(action, dataset, dataset_description)
    generate_description(dataset, action)
    train = list(
        filter(lambda split: split.name == dataset.train_split_name, dataset.splits)
    )[0]
    test = list(
        filter(lambda split: split.name == 'test_merged', dataset.splits)
    )[0]
    responses = columns_clarification(action, dataset_description)
    prediction = predict(train, test, responses)
    pprint(
        accuracy_score(dataset.splits[0].data[responses["target_column"]], prediction)
    )


def load_data(dataset_name):
    logger.info(f"===Loading dataset {dataset_name}===")
    dataset_path = os.sep.join([dataset_name])
    dataset = Dataset.load_from_path(dataset_path)
    return dataset


def data_clarification(action, dataset, dataset_descriptions):
    logger.info("===Clarifying dataset===")
    task_prompts = {
        "dataset_name": {
            "system": dataset_descriptions,
            "task": prompts.dataset_name_prompt,
            "context": None,
        },
        "train_split": {
            "system": dataset_descriptions,
            "task": prompts.train_split_definition_prompt,
            "context": dataset.get_description(),
        },
    }

    responses = action.run_model_multicall(task_prompts)
    operations = {"train_split": lambda x: x.split(".")[0]}
    responses = action.process_model_responses(responses, operations)

    dataset.name = responses["dataset_name"]
    dataset.train_split_name = responses["train_split"]


def generate_description(dataset, action):
    logger.info("===Generating column descriptions===")
    train = list(
        filter(lambda split: split.name == dataset.train_split_name, dataset.splits)
    )[0]
    column_descriptions = action.generate_all_column_description(
        split=train, dataset=dataset
    )
    train.set_column_descriptions(column_descriptions)


def columns_clarification(action, dataset_description):
    logger.info("===Clarifying columns===")
    task_prompts = {
        "categorical_columns": {
            "system": dataset_description,
            "task": prompts.categorical_definition_prompt,
            "context": prompts.categorical_definition_context,
        },
        "target_column": {
            "system": dataset_description,
            "task": prompts.target_definition_prompt,
            "context": None,
        },
        "task_type": {
            "system": dataset_description,
            "task": prompts.task_definition_prompt,
            "context": None,
        },
    }

    responses = action.run_model_multicall(task_prompts)

    pattern = r"[\'\"“”‘’`´]"
    operations = {
        "categorical_columns": lambda x: x.split("\n"),
        "target_column": lambda x: re.sub(pattern, "", x),
        "task_type": lambda x: re.sub(pattern, "", x.lower()),
    }
    responses = action.process_model_responses(responses, operations)
    return responses


def predict(train, test, responses):
    logger.info("===Processing prediction===")
    prediction = run_example(
        train_df=train.data,
        test_df=test.data,
        problem=responses["task_type"],
        target=responses["target_column"],
    )
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("dataset_description", type=str, help="Dataset description")

    args = parser.parse_args()
    main(args.dataset_name, args.dataset_description)
