import ast
import json
import logging
import os
import random
import re
from typing import Dict

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
)

from fedot_llm.data.data import Dataset, Split
from fedot_llm.language_models.base import BaseLLM

_MAX_RETRIES = 6

logger = logging.getLogger(__name__)

class ModelAction():

    def __init__(self, model) -> None:
        self.model = model

    def run_model_multicall(self, tokenizer, generation_config, prompts):
        """Run all prompts on local model
        """

        responses = {}
        for task in prompts:
            messages = [
                {"role": "system", "content": prompts[task]["system"]},
                {"role": "context", "content": prompts[task]["context"]},
                {"role": "user", "content": prompts[task]["task"]},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                input_ids,
                **generation_config
            )
            response = outputs[0][input_ids.shape[-1]:]
            responses[task] = tokenizer.decode(
                response, skip_special_tokens=True)

        return responses

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=(retry_if_exception_type(RequestException)
               | retry_if_exception_type(RuntimeError)),
        reraise=True
    )
    def run_web_model_multicall(self, prompts):
        """Run all prompts on web model
        """

        responses = {}
        for task in prompts:
            self.model.set_sys_prompt(prompts[task]["system"])
            self.model.add_context(prompts[task]["context"])

            response = self.model(prompts[task]["task"], as_json=True)
            responses[task] = response

        return responses

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

        sys_prompt = """You are helpful AI assistant.
        User will enter one column from dataset, and the assistant will make one sentence discription of data in this column.
        Don't make assumptions about what values to plug into functions. Use column hint.
        Output format: only JSON using the schema defined here: {schema}""".format(schema=json.dumps(schema))

        user_template = """Dataset Title: {title}
        Dataset description: {ds_descr}
        Column name: {col_name}
        Column hint: {hint}
        Column values: 
        ```
        {values}
        ```
        """

        self.model.set_context(column.head(30).to_markdown(index=False))
        column_uniq_vals = column.unique().tolist()
        column_vals = pd.Series(column_uniq_vals if len(
            column_uniq_vals) < 30 else random.sample(column_uniq_vals, k=30), name=column.name)
        user_prompt = user_template.format(
            title=dataset.name,
            ds_descr=dataset.description,
            col_name=column.name,
            hint=split.get_column_hint(column.name),
            values=column_vals.to_markdown(index=False)
        )
        response = self.model(user_prompt, as_json=True)
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

        schema = {
            "data": {
                "type": "string",
                "description": "one line plain text"
            }
        }

        sys_prompt = """You are helpful AI assistant.
        User will enter one column from dataset, and the assistant will make one sentence discription of data in this column.
        Don't make assumptions about what values to plug into functions. Use column hint.
        Output format: only JSON using the schema defined here: {schema}""".format(schema=json.dumps(schema))

        self.model.set_sys_prompt(sys_prompt)

        result = {}

        for col_name in split.data.columns:
            result[col_name] = self.__generate_column_description(column=split.data[col_name],
                                                                  split=split,
                                                                  dataset=dataset)['data']['description']
        return result

    def process_model_responses(responses):
        responses["categorical_columns"] = responses["categorical_columns"].split(
            "\n")
        responses["task_type"] = responses["task_type"].lower()
        return responses

    def save_model_responses(responses, path):
        with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
            json.dump(responses, json_file)
