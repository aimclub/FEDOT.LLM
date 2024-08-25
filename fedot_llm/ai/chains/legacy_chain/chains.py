"""
THIS MODULE IS DEPRICATED AND WILL BE REMOVED IN THE FUTURE
NOT USE IT FOR NEW DEVELOPMENT
"""

import logging
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, List, Literal, Optional

from fedot.api import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.dag.graph_utils import graph_structure
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import (BaseOutputParser, JsonOutputParser,
                                           PydanticOutputParser,
                                           StrOutputParser)
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (Runnable, RunnableLambda,
                                      RunnableParallel, RunnablePassthrough,
                                      RunnablePick)
from numpy import ndarray
from pydantic import BaseModel, Field

from fedot_llm import prompts
from fedot_llm.data import Dataset
from langchain.output_parsers import OutputFixingParser
from nlangchain.output_parsers.retry import \
    RetryWithErrorOutputParser  # My PR already in master of langchain but not in pypi yet



class ColumnType(BaseModel):
    name: str = Field(description="The name of the colum")
    column_type: Literal["categorical", "numerical"] = Field(
        description="The variables type: categorical or numerical"
    )

@dataclass
class Step:
    id: str
    name: str
    status: Literal['Waiting', 'Running', 'Streaming',
    'Сompleted'] = field(default='Waiting')

    def __str__(self):
        return self.name


steps: List[Step] = [
    Step('dataset_name_chain', 'Define Dataset Name'),
    Step('dataset_description_chain', 'Define Dataset Description'),
    Step('dataset_goal_chain', 'Define Dataset Goal'),
    Step('dataset_train_chain', 'Define Train Split'),
    Step('dataset_test_chain', 'Define Test Split'),
    Step('dataset_target_chain', 'Define Target Column'),
    Step('dataset_task_type_chain', 'Define Task Type'),
    Step('categorize_runnable',
         'Create Column Descriptions And Define Columns Category'),
    Step('fedot_predict', 'Fedot makes predictions'),
    Step('fedot_analyze_predictions_chain', 'Fedot Analyze Results')
]


@dataclass
class FedotPredictions:
    predictions: ndarray[Any, Any]
    auto_model: Fedot
    best_pipeline: Pipeline


@dataclass
class ChainBuilder:
    assistant: BaseChatModel
    """Model that is answering on requests."""
    dataset: Dataset
    """The dataset that is used in analysis."""
    arbiter: Optional[BaseChatModel] = None
    """Model that can be used to check and correct the assistant's output."""
    retry_num: int = 10
    """Number of retries to get the correct output."""

    def __post_init__(self):
        if self.arbiter is None:
            self.arbiter = self.assistant

    def with_retry_chain(self, parser: BaseOutputParser, prompt: Optional[PromptValue] = None):
        fixing_retry_parser = OutputFixingParser.from_llm(
            parser=parser, llm=(self.arbiter or self.assistant),
            max_retries=3
        )
        error_retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=fixing_retry_parser, llm=(self.arbiter or self.assistant),
            max_retries=3
        )
        return (
                {
                    "completion": self.assistant.with_retry(wait_exponential_jitter=True,
                                                            stop_after_attempt=self.retry_num) | StrOutputParser(),
                    "prompt_value": lambda x: PromptTemplate.from_template('{instractions}').invoke(
                        {'instractions': (prompt or parser.get_format_instructions())})
                }
                | RunnableLambda(lambda x: error_retry_parser.parse_with_prompt(x["completion"], x["prompt_value"]))
                .with_retry(wait_exponential_jitter=True, stop_after_attempt=self.retry_num)
        ).with_config({'tags': ['retry']})


    @property
    def describe_column_chain(self):
        """Chain for describing a column

        INPUT:
            column_name -- the name of the column
        """
        parser = PydanticOutputParser(pydantic_object=ColumnDescription)

        describe_column_template = prompts.describe_column_template.partial(
            dataset_title=self.dataset.name,
            dataset_description=self.dataset.description,
            format_instructions=parser.get_format_instructions(),
        )
        processing_chain = (
                describe_column_template
                | self.with_retry_chain(parser=JsonOutputParser(pydantic_object=ColumnDescription))
        )

        return (
                {
                    "column_name": lambda x: x,
                    "column_samples": lambda x: self.dataset.train_split[x]
                .head(10)
                .to_markdown(index=False),
                }
                | RunnablePassthrough.assign(column_description=processing_chain)
        ).with_config({"run_name": "describe_column_chain"})

    @property
    def categorize_column_chain(self):
        """Chain for categorizing a column

        INPUT:
            column_name -- the name of the column
            column_samples -- samples of the column
            column_description -- the ColumnDescription object
        """

        parser = PydanticOutputParser(pydantic_object=ColumnType)

        categorical_columns_def = prompts.categorical_template.partial(
            dataset_title=self.dataset.name,
            dataset_description=self.dataset.description,
            format_instructions=parser.get_format_instructions(),
        )

        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=parser, llm=(self.arbiter or self.assistant)
        )
        return (
                RunnablePassthrough.assign(
                    column_description=lambda x: x["column_description"]["description"]
                ).assign(
                    column_ratio=lambda x: round(
                        self.dataset.train_split[x["column_name"]].nunique()
                        / len(self.dataset.train_split[x["column_name"]].dropna()),
                        2,
                    )
                )
                | categorical_columns_def
                | {
                    "completion": self.assistant.with_retry(wait_exponential_jitter=True,
                                                            stop_after_attempt=self.retry_num) | StrOutputParser(),
                    "prompt_value": lambda x: prompts.categorical_fix_template.invoke({})

                }
                | {
                    "reasoning": itemgetter("completion"),
                    "category": RunnableLambda(lambda x: retry_parser
                                               .parse_with_prompt(x["completion"], x["prompt_value"]))
                .with_retry(wait_exponential_jitter=True, stop_after_attempt=self.retry_num)
                }
        ).with_config({"run_name": "categorize_column_chain"})

    @property
    def categorize_columns_chain(self):
        """Chain for categorizing columns

        INPUT:
            column_name -- the name of the column
        """

        return self.describe_column_chain | self.categorize_column_chain

    def split_train_to_2_splits(self, input):
        from sklearn.model_selection import train_test_split
        new_train, new_test = train_test_split(input['train_split'].data, train_size=0.8, random_state=42)
        return {'new_train': new_train, 'new_test': new_test}

    def __categorize_runnable(self, x):
        return self.categorize_columns_chain.batch(x)

    def fedot_predict_runnable(self, input):
        auto_model = Fedot(problem=input['dataset_task_type'],
                           seed=42,
                           timeout=1,
                           cv_folds=10,
                           with_tuning=True,
                           metric=['roc_auc', 'accuracy'],
                           logging_level=logging.FATAL,
                           n_jobs=-1)
        best_pipeline = auto_model.fit(
            features=input['new_splits']['new_train'], target=input['dataset_target'])
        predictions = auto_model.predict(features=input['new_splits']['new_test'])
        return {'predictions': predictions, 'auto_model': auto_model, 'best_pipeline': best_pipeline}

    @property
    def fedot_analyze_predictions_chain(self):
        return (
                RunnablePassthrough
                .assign(parameters=lambda input: graph_structure(input['fedot']['best_pipeline']))
                .assign(metrics=lambda input: input['fedot']['auto_model'].get_metrics())
                | prompts.analyze_predictions
                | self.assistant
                | StrOutputParser().with_config({"tags": ["print"]})
        ).with_config({"run_name": "fedot_analyze_predictions_chain"})

    @property
    def predict_chain(self):

        return (
                self.dataset_metadata_chain
                | RunnablePassthrough.assign(train_split=lambda input:
        list(filter(lambda split: split.name == input['dataset_train'], self.dataset.splits))[0])
                | RunnablePassthrough.assign(test_split=lambda input:
        list(filter(lambda split: split.name == input['dataset_test'], self.dataset.splits))[0])
                | RunnablePassthrough.assign(train_split_columns=lambda input: list(input['train_split'].data.columns))
                | RunnablePassthrough.assign(
            categorize=(itemgetter("train_split_columns") | RunnableLambda(
                self.__categorize_runnable)).with_config({"run_name": "categorize_runnable"})
        )
                | RunnablePassthrough.assign(new_splits=self.split_train_to_2_splits)
                | RunnablePassthrough.assign(
            fedot=RunnableLambda(self.fedot_predict_runnable).with_config({"run_name": "fedot_predict"}))
                | RunnablePassthrough.assign(analyze=self.fedot_analyze_predictions_chain)
                | RunnablePick('fedot')
                | RunnableLambda(lambda input: FedotPredictions(predictions=input['predictions'],
                                                                auto_model=input['auto_model'],
                                                                best_pipeline=input['best_pipeline']))
        ).with_config({"run_name": "master"})

    @property
    def master_chain(self):
        """Chain for categorizing columns

        INPUT:
            big_description -- user input big description of dataset
        """
        return (
                self.dataset_metadata_chain
                | RunnablePassthrough.assign(train_split=lambda input:
        list(filter(lambda split: split.name == input['dataset_train'], self.dataset.splits))[0])
                | RunnablePassthrough.assign(test_split=lambda input:
        list(filter(lambda split: split.name == input['dataset_test'], self.dataset.splits))[0])
                | RunnablePassthrough.assign(train_split_columns=lambda input: list(input['train_split'].data.columns))
                | RunnablePassthrough.assign(
            categorize=(itemgetter("train_split_columns") | RunnableLambda(
                self.__categorize_runnable)).with_config({"run_name": "categorize_runnable"})
        )
        )
