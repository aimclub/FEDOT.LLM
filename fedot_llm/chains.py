from ast import parse
from operator import itemgetter
import re
from typing import Literal, Optional
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
    JsonOutputParser,
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from pydantic import BaseModel, Field

from fedot_llm import prompts
from fedot_llm.data import Dataset

from fedot_llm.data.data import Split
from nlangchain.output_parsers.retry import (
    RetryWithErrorOutputParser,
)  # My PR already in master of langchain but not in pypi yet


class ColumnDescription(BaseModel):
    name: str = Field(description="The name of the column")
    description: str = Field(description="The short description of the column")


class ColumnType(BaseModel):
    name: str = Field(description="The name of the colum")
    column_type: Literal["categorical", "numerical"] = Field(
        description="The variables type: categorical or numerical"
    )


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
        )

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
        )

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
            | (lambda x: setattr(self.dataset, "target", x) or x)
        )

    def __clear_and_set_split(
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
            | (lambda name: self.__clear_and_set_split(str(name), "train") or name)
        )

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
            | (lambda name: self.__clear_and_set_split(str(name), "test") or name)
        )

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
            | (lambda x: re.sub(r"[\'\"“”‘’`´]", "", x) or x)
            | (lambda x: setattr(self.dataset, "target_name", x) or x)
        )

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
            | (lambda x: re.sub(r"[\'\"“”‘’`´]", "", x.lower()))
            | (lambda x: setattr(self.dataset, "task_type", x) or x)
        )

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
        )

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
            | self.assistant
            | JsonOutputParser(pydantic_object=ColumnDescription)
        )

        return {
            "column_name": lambda x: x,
            "column_samples": lambda x: self.dataset.train_split[x]
            .head(10)
            .to_markdown(index=False),
        } | RunnablePassthrough.assign(column_description=processing_chain)

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

        )

    @property
    def categorize_columns_chain(self):
        """Chain for categorizing columns

        INPUT:
            column_name -- the name of the column
        """

        return self.describe_column_chain | self.categorize_column_chain
