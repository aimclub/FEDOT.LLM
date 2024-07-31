import contextlib
import json
import logging
import os
import re
from typing import Dict

from tenacity import retry, stop_after_attempt

from fedot_llm.data.data import Dataset, Split
from fedot_llm.language_models.base import BaseLLM
from fedot_llm.language_models.prompts import (
    categorical_sys_template,
    categorical_user_template,
    describe_column_sys,
    describe_column_user,
)

_MAX_RETRIES = 6
COLUMN_DESCR_RESPONSE = re.compile(
    r"\{\s*['\"]\s*data\s*['\"]\s*:\s*\{\s*['\"]type['\"]\s*:\s*['\"]string['\"]\s*\,\s*['\"]description['\"]\s*:\s*['\"].*['\"]\s*\}\s*\}"
)

COLUMN_CATEGORY_RESPONSE = re.compile(r"\{\s*['\"]\s*data\s*['\"]\s*:\s*['\"]\s*(yes|not)\s*['\"]\s*\}")



logger = logging.getLogger(__name__)


@contextlib.contextmanager
def log_and_raise(message, level=logging.ERROR):
    try:
        yield
    except RuntimeError as e:
        logger.log(level, f"{message}: {str(e)}")
        raise RuntimeError(f"{message}: {str(e)}")


class ModelAction:
    def __init__(self, model: BaseLLM) -> None:
        self.model = model

    def run_model_call(self, system, context, task, **kwargs):
        """Run a prompt on model"""
        response = self.model.generate(
            user_prompt=task, sys_prompt=system, context=context
        )
        return response

    def run_model_multicall(self, prompts):
        """Run a list of prompts on web model"""
        responses = {}
        for task in prompts:
            response = self.run_model_call(
                system=prompts[task]["system"],
                context=prompts[task]["context"],
                task=prompts[task]["task"],
            )
            responses[task] = response
        return responses

    @staticmethod
    def process_model_responses(responses, operations):
        for key in operations:
            responses[key] = operations[key](responses[key])
        return responses

    @classmethod
    def process_model_responses_for_v1(cls, responses):
        operations = {
            "categorical_columns": lambda x: x.split("\n"),
            "task_type": lambda x: x.lower(),
        }
        responses = cls.process_model_responses(responses, operations)
        return responses

    @staticmethod
    def save_model_responses(responses, path):
        with open(os.sep.join([path, "model_responses.json"]), "w") as json_file:
            json.dump(responses, json_file)

    @retry(stop=stop_after_attempt(_MAX_RETRIES), reraise=True)
    def __generate_column_description(
        self, column_name: str, split: Split, dataset: Dataset
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate a description for a column.

        Args:
        - column_name (str): The column name for which to generate a description.
        - split (Split): The split containing the column.
        - dataset (Dataset): The dataset containing the column.

        Returns:
        dict: A dictionary containing the generated description for the column.

        Raises:
        RuntimeError: If the answer is not found in the response or if the 'data' node is not found in the response.
        """

        schema = {"data": {"type": "string", "description": "one line plain text"}}

        sys_prompt = describe_column_sys.format(schema=json.dumps(schema))

        column_vals = split.get_unique_values(column_name, max_number=30)
        user_prompt = describe_column_user.format(
            title=dataset.name,
            ds_descr=dataset.description,
            col_name=column_name,
            hint=split.get_column_hint(column_name),
            values=column_vals.to_markdown(index=False),
        )

        response = self.run_model_call(
            task=user_prompt,
            system=sys_prompt,
            context=split.data[column_name].head(10).to_markdown(index=False),
        )
        response = re.sub(r"\s+", " ", response.strip())
        re.sub(r"['\"]\s*([\w+])\s*['\"]", '"\1"', response)

        with log_and_raise(
            f"Description not found in: {response}", level=logging.WARNING
        ):
            answer = re.findall(COLUMN_DESCR_RESPONSE, response)
            if not answer:
                raise ValueError(f"Answer not found in response {response}")

        with log_and_raise(
            f"Data node not found in: {answer[0]}", level=logging.WARNING
        ):
            dict_resp = json.loads(answer[0])
            if "data" not in dict_resp:
                raise ValueError(f"Data not found in dict {dict_resp}")
        return dict_resp

    def generate_all_column_description(
        self, split: Split, dataset: Dataset
    ) -> Dict[str, str]:
        """Generate descriptions for all columns in the provided table.

        Args:
            split (pd.DataFrame): A split representing the table with columns to describe.
            dataset (Dataset): The dataset used for generating column descriptions.

        Returns:
            A dictionary where keys are column names and values are descriptions generated for each column.
        """

        result = {}

        for col_name in split.data.columns:
            result[col_name] = self.__generate_column_description(
                column_name=col_name, split=split, dataset=dataset
            )["data"]["description"]
        return result

    @retry(stop=stop_after_attempt(_MAX_RETRIES), reraise=True)
    def __get_categorical_feature(self, column_name, split: Split, dataset: Dataset):
        schema = {
            "data": "yes | not"
        }
        sys_prompt = categorical_sys_template.format(schema=json.dumps(schema))
        user_prompt = categorical_user_template.format(
            title=dataset.name,
            ds_desc=dataset.description,
            col_name=column_name,
            col_desc=(split.column_descriptions.get(column_name, "")),
            col_ratio=split.unique_ratios.get(column_name, ""),
            data=split.get_unique_values(column_name=column_name, max_number=30).to_markdown(index=False)
        )
        
        context_prompt = split.data[column_name].head(10).to_markdown(index=False)
        response = self.run_model_call(
            task=user_prompt,
            system=sys_prompt,
            context=context_prompt
        )
        response = re.sub(r"\s+", " ", response.strip().replace('\n', '').lower())
        with log_and_raise(
            f"Category not found in: {response}", level=logging.WARNING
        ):
            answer = re.findall(COLUMN_CATEGORY_RESPONSE, response)
            if not answer:
                raise ValueError(f"Answer not found in response {response}")
        return answer[0]
    
    def get_categorical_features(self, split: Split, dataset: Dataset):
        result = []
        for col_name in split.data.columns:
            is_categorical= self.__get_categorical_feature(
                column_name=col_name, split=split, dataset=dataset
            )
            if is_categorical == 'yes':
                result.append(col_name)
            elif is_categorical == 'not':
                pass
            else:
                raise ValueError(f"Unexpected response: {is_categorical}")
                
        return result
        
        


# def run_model_multicall(model, tokenizer, generation_config, prompts):
#     """Run all prompts on local model

#     TODO: transform to an interaction with local model helper class
#     """

#     responses = {}
#     for task in prompts:
#         messages = [
#             {"role": "system", "content": prompts[task]["system"]},
#             {"role": "context", "content": prompts[task]["context"]},
#             {"role": "user", "content": prompts[task]["task"]},
#         ]

#         input_ids = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(model.device)

#         outputs = model.generate(
#             input_ids,
#             **generation_config
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         responses[task] = tokenizer.decode(response, skip_special_tokens=True)

#     return responses
