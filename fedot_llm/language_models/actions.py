import json
import logging
import os
import random
import re
from typing import Dict

import pandas as pd
from tenacity import retry, stop_after_attempt

from fedot_llm.data.data import Dataset, Split
from fedot_llm.language_models.base import BaseLLM
from fedot_llm.language_models.prompts import (describe_column_sys,
                                               describe_column_user)

_MAX_RETRIES = 6

logger = logging.getLogger(__name__)

class ModelAction():
    
    def __init__(self, model: BaseLLM) -> None:
        self.model = model

    def run_model_call(self, system, context, task):
        """Run a prompt on model
        """
        response = self.model.generate(user_prompt=task,
                                       sys_prompt=system,
                                       context=context)
        return response

    def run_model_multicall(self, prompts):
        """Run a list of prompts on web model
        """
        responses = {}
        for task in prompts:
            response = self.run_model_call(
                system = prompts[task]["system"],
                context = prompts[task]["context"],
                task = prompts[task]["task"]
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
            "categorical_columns": lambda x : x.split("\n"),
            "task_type": lambda x : x.lower()
        }
        responses = cls.process_model_responses(responses, operations)
        return responses

    @staticmethod
    def save_model_responses(responses, path):
        with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
            json.dump(responses, json_file)
            
    @retry(
        stop=stop_after_attempt(_MAX_RETRIES),
        reraise=True
    )
    def __generate_column_description(self, column: pd.Series, split: Split, dataset: Dataset):
        """
        Generate a description for a column.

        Args:
        - column (pd.Series): The column for which the description is generated.
        - dataset (Dataset): The dataset containing the column.

        Returns:
        dict: A dictionary containing the generated description for the column.

        Raises:
        RuntimeError: If the answer is not found in the response or if the 'data' node is not found in the response.
        """
        FIND_ANSWER = re.compile(
            r"\{\s*['\"]\s*data\s*['\"]\s*:\s*\{\s*['\"]type['\"]\s*:\s*['\"]string['\"]\s*\,\s*['\"]description['\"]\s*:\s*['\"].*['\"]\s*\}\s*\}")

        
        schema = {
            "data": {
                "type": "string",
                "description": "one line plain text"
            }
        }

        sys_prompt = describe_column_sys.format(schema=json.dumps(schema))
        
        column_uniq_vals = column.unique().tolist()
        column_vals = pd.Series(column_uniq_vals if len(
            column_uniq_vals) < 30 else random.sample(column_uniq_vals, k=30), name=column.name)
        user_prompt = describe_column_user.format(
            title=dataset.name,
            ds_descr=dataset.description,
            col_name=column.name,
            hint=split.get_column_hint(column.name),
            values=column_vals.to_markdown(index=False)
        )
        
        response = self.run_model_call(task=user_prompt,
                                       system=sys_prompt,
                                       context=column.head(30).to_markdown(index=False))
        response = response.strip().replace('\n', '').capitalize()
        response = ' '.join(response.split())
        re.sub(r"['\"]\s*([\w+])\s*['\"]", '"\1"', response)

        answer = re.findall(FIND_ANSWER, response)
        if not answer:
            logger.warning("Description not found in: ", response)
            raise RuntimeError("Answer not found in: ", response)

        dict_resp = json.loads(answer[0])
        if "data" not in dict_resp:
            raise RuntimeError("Data node not found in: ", response)
        return dict_resp

    def generate_all_column_description(self, split: Split, dataset: Dataset) -> Dict[str, str]:
        """ Generate descriptions for all columns in the provided table.

        Args:
            split (pd.DataFrame): A split representing the table with columns to describe.
            dataset (Dataset): The dataset used for generating column descriptions.

        Returns:
            A dictionary where keys are column names and values are descriptions generated for each column.
        """

        result = {}

        for col_name in split.data.columns:
            result[col_name] = self.__generate_column_description(column=split.data[col_name],
                                                                  split=split,
                                                                  dataset=dataset)['data']['description']
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
