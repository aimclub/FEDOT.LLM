import logging
import re
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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (Runnable, RunnableAssign, RunnableLambda,
                                      RunnableParallel, RunnablePassthrough,
                                      RunnablePick, RunnableSequence,
                                      RunnableSerializable)
from numpy import ndarray
from pydantic import BaseModel, Field

from fedot_llm import prompts
from fedot_llm.data import Dataset
from fedot_llm.data.data import Split
from nlangchain.output_parsers.retry import \
    RetryWithErrorOutputParser  # My PR already in master of langchain but not in pypi yet


class ColumnDescription(BaseModel):
    name: str = Field(description="The name of the column")
    description: str = Field(description="The short description of the column")


class ColumnType(BaseModel):
    name: str = Field(description="The name of the colum")
    column_type: Literal["categorical", "numerical"] = Field(
        description="The variables type: categorical or numerical"
    )


@dataclass
class Stage:
    name: str
    display_name: str
    status:  Literal['Waiting', 'Running', 'Streaming',
                     'Сompleted'] = field(default='Waiting')

    def __str__(self):
        return self.name


stages: List[Stage] = [
    Stage('dataset_name_chain', 'Define Dataset Name'),
    Stage('dataset_description_chain', 'Define Dataset Description'),
    Stage('dataset_goal_chain', 'Define Dataset Goal'),
    Stage('dataset_train_chain', 'Define Train Split'),
    Stage('dataset_test_chain', 'Define Test Split'),
    Stage('dataset_target_chain', 'Define Target Column'),
    Stage('dataset_task_type_chain', 'Define Task Type'),
    Stage('categorize_runnable',
          'Create Column Descriptions And Define Columns Category'),
    Stage('fedot_predict', 'Fedot makes predictions'),
    Stage('fedot_analyze_predictions_chain', 'Fedot Analyze Results')
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
        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=parser, llm=(self.arbiter or self.assistant)
        )
        return (
            {
                "completion": self.assistant.with_retry(wait_exponential_jitter=True, stop_after_attempt=self.retry_num) | StrOutputParser(),
                "prompt_value": lambda x: PromptTemplate.from_template('{instractions}').invoke({'instractions': (prompt or parser.get_format_instructions())})
            }
            | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x["completion"], x["prompt_value"]))
            .with_retry(wait_exponential_jitter=True, stop_after_attempt=self.retry_num)
        )

    @property
    def dataset_name_chain(self):
        """Chain for defining and setting the dataset name

        INPUT:
        - big_description -- user input big description of dataset
        """
        return (
            prompts.dataset_name_template
            | self.assistant
            | StrOutputParser()
            | (lambda x: setattr(self.dataset, "name", x) or x)
        ).with_config({"run_name": "dataset_name_chain"})

    @property
    def dataset_description_chain(self):
        """Chain for defining and setting the dataset description

        INPUT:
        - big_description -- user input big description of dataset
        """
        return (
            prompts.dataset_description_template
            | self.assistant
            | StrOutputParser()
            | (lambda x: setattr(self.dataset, "description", x) or x)
        ).with_config({"run_name": "dataset_description_chain"})

    @property
    def dataset_goal_chain(self):
        """Chain for defining and setting the dataset goal

        INPUT:
            big_description -- user input big description of dataset
        """
        return (
            prompts.dataset_goal_template
            | self.assistant
            | StrOutputParser()
            | (lambda x: setattr(self.dataset, "goal", x) or x)
        ).with_config({"run_name": "dataset_goal_chain"})

    def __set_split(
        self, split_name: str, split_type: Literal["train", "test"]
    ):
        split_name = split_name.split(".")[0]
        if split_type == "train":
            self.dataset.train_split = split_name
        elif split_type == "test":
            self.dataset.test_split = split_name
        else:
            raise ValueError(f"Not supported split type: {split_type}")

    @property
    def dataset_train_chain(self):
        """Chain for defining and setting the dataset train split

        INPUT:
            detailed_description: property of the dataset object
        """
        return (
            prompts.train_split_template
            | self.assistant
            | StrOutputParser()
            | (lambda name: name.split(".")[0].strip().strip(r",.\'\"“”‘’`´"))
            | (lambda name: self.__set_split(str(name), "train") or name)
        ).with_config({"run_name": "dataset_train_chain"})

    @property
    def dataset_test_chain(self):
        """Chain for defining and setting the dataset test split

        INPUT:
            detailed_description: property of the dataset object
        """
        return (
            prompts.test_split_template
            | self.assistant
            | StrOutputParser()
            | (lambda name: name.split(".")[0].strip().strip(r",.\'\"“”‘’`´"))
            | (lambda name: self.__set_split(str(name), "test") or name)
        ).with_config({"run_name": "dataset_test_chain"})

    @property
    def dataset_target_chain(self):
        """Chain for defining and setting the dataset target column

        INPUT:
            detailed_description: property of the dataset object
        """
        return (
            prompts.target_definition_template
            | self.assistant
            | StrOutputParser()
            | (lambda x: x.strip().strip(r",.\'\"“”‘’`´"))
            | (lambda x: setattr(self.dataset, "target_name", x) or x)
        ).with_config({"run_name": "dataset_target_chain"})

    @property
    def dataset_task_type_chain(self):
        """Chain for defining and setting the dataset task type

        INPUT:
            detailed_description: property of the dataset object
        """
        return (
            prompts.task_definition_template
            | self.assistant
            | StrOutputParser()
            | (lambda x: x.lower().strip().strip(r",.\'\"“”‘’`´"))
            | (lambda x: setattr(self.dataset, "task_type", x) or x)
        ).with_config({"run_name": "dataset_task_type_chain"})

    @property
    def dataset_metadata_chain(self) -> Runnable:
        """Chain for defining and setting the dataset metadata

        INPUT:
            big_description -- user input big description of dataset
        """
        return (
            RunnableParallel(
                dataset_name=self.dataset_name_chain,
                dataset_description=self.dataset_description_chain,
                dataset_goal=self.dataset_goal_chain,
            )
            | RunnablePassthrough.assign(detailed_description=lambda _: self.dataset.detailed_description)
            | RunnableParallel(
                dataset_info=RunnablePassthrough(),
                dataset_train=self.dataset_train_chain,
                dataset_test=self.dataset_test_chain,
                dataset_target=self.dataset_target_chain,
                dataset_task_type=self.dataset_task_type_chain,
            )
        ).with_config({"run_name": "dataset_metadata_chain"})

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
                "completion": self.assistant.with_retry(wait_exponential_jitter=True, stop_after_attempt=self.retry_num) | StrOutputParser(),
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

    def categorize_runnable(self, x):
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
            | RunnablePassthrough.assign(train_split=lambda input: list(filter(lambda split: split.name == input['dataset_train'], self.dataset.splits))[0])
            | RunnablePassthrough.assign(test_split=lambda input: list(filter(lambda split: split.name == input['dataset_test'], self.dataset.splits))[0])
            | RunnablePassthrough.assign(train_split_columns=lambda input: list(input['train_split'].data.columns))
            | RunnablePassthrough.assign(
                categorize=(itemgetter("train_split_columns") | RunnableLambda(
                    self.categorize_runnable)).with_config({"run_name": "categorize_runnable"})
            )
            | RunnablePassthrough.assign(new_splits=self.split_train_to_2_splits)
            | RunnablePassthrough.assign(fedot=RunnableLambda(self.fedot_predict_runnable).with_config({"run_name": "fedot_predict"}))
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
            | RunnablePassthrough.assign(train_split=lambda input: list(filter(lambda split: split.name == input['dataset_train'], self.dataset.splits))[0])
            | RunnablePassthrough.assign(test_split=lambda input: list(filter(lambda split: split.name == input['dataset_test'], self.dataset.splits))[0])
            | RunnablePassthrough.assign(train_split_columns=lambda input: list(input['train_split'].data.columns))
            | RunnablePassthrough.assign(
                categorize=(itemgetter("train_split_columns") | RunnableLambda(
                    self.categorize_runnable)).with_config({"run_name": "categorize_runnable"})
            )
        )
