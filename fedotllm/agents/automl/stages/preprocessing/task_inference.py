import logging
from typing import Any, Dict, List, Optional

from fedotllm.constants import (
    CLASSIFICATION_PROBLEM_TYPES,
    METRICS_BY_PROBLEM_TYPE,
    METRICS_DESCRIPTION,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    PROBLEM_TYPES,
    TASK_TYPES,
)
from fedotllm.exceptions import OutputParserException
from fedotllm.heuristics import infer_problem_type
from fedotllm.llm import LiteLLMModel
from fedotllm.predictor.task import PredictionTask
from fedotllm.prompting.generators import (
    DataFilePromptGenerator,
    EvalMetricPromptGenerator,
    JsonFieldPromptGenerator,
    LabelColumnPromptGenerator,
    OutputIDColumnPromptGenerator,
    ProblemTypePromptGenerator,
    TaskTypePromptGenerator,
    TestIDColumnPromptGenerator,
    TrainIDColumnPromptGenerator,
)

logger = logging.getLogger(__name__)


class TaskInference:
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    def __init__(self, llm: LiteLLMModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm: LiteLLMModel = llm
        self.fallback_value = None
        self.ignored_value: List[str] = []

    def initialize_task(self, task: PredictionTask):
        self.prompt_genetator: Optional[JsonFieldPromptGenerator] = None
        self.valid_values = None

    def log_value(self, key: str, value: Any, max_width: int = 1600) -> None:
        """Logs a key-value pair with formatted output"""
        if not value:
            logger.info(
                f"WARMING: Failed to identify the {key} of the task, it is set to None."
            )
            return

        prefix = key
        value_str = str(value).replace("\n", "\\n")
        if len(prefix) + len(value_str) > max_width:
            value_str = value_str[: max_width - len(prefix) - 3] + "..."

        bold_start = "\033[1m"
        bold_end = "\033[0m"

        logger.info(f"{bold_start}{prefix}{bold_end}: {value_str}")

    def transform(self, task: PredictionTask) -> PredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        for k, v in parser_output.items():
            if v in self.ignored_value:
                v = None
            self.log_value(k, v)
            setattr(task, k, self.post_process(task=task, value=v))
        return task

    def post_process(self, task, value):
        return value

    def _chat_and_parse_prompt_output(self) -> Dict[str, str]:
        try:
            assert self.prompt_genetator is not None, (
                "prompt_generator is not initialized"
            )
            chat_prompt = self.prompt_genetator.chat_prompt
            logger.debug(f"LLM chat_prompt:\n{chat_prompt}")
            output = self.llm.query(chat_prompt)
            logger.debug(f"LLM output:\n{output}")
            parsed_output = self.prompt_genetator.create_parser()(
                output,
                valid_values=self.valid_values,
                fallback_value=self.fallback_value,
            )
            return parsed_output
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(str(self.llm.stats))
            raise e


class DataFileNameInference(TaskInference):
    """Uses an LLM to locate the filenames of the train, test, and output data,
    and assigns them to the respective properties of the task.
    """

    def initialize_task(self, task):
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.ignored_value = [NO_FILE_IDENTIFIED]
        self.prompt_genetator = DataFilePromptGenerator(
            data_description=task.metadata["description"], filenames=filenames
        )


class TaskTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = TASK_TYPES
        self.prompt_genetator = TaskTypePromptGenerator(
            data_description=task.metadata["description"]
        )


class LabelColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.train_data.columns)
        self.valid_values = column_names
        self.prompt_genetator = LabelColumnPromptGenerator(
            data_description=task.metadata["description"], column_names=column_names
        )


class ProblemTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = PROBLEM_TYPES
        self.prompt_genetator = ProblemTypePromptGenerator(
            data_description=task.metadata["description"]
        )

    def post_process(self, task, value):
        # LLM may get confused between BINARY and MULTICLASS as it cannot see the whole label column
        if value in CLASSIFICATION_PROBLEM_TYPES:
            problem_type_infered_by_autogloun = infer_problem_type(
                task.train_data[task.label_column], silent=True
            )
            if problem_type_infered_by_autogloun in CLASSIFICATION_PROBLEM_TYPES:
                value = problem_type_infered_by_autogloun
        return value


class BaseIDColumnInference(TaskInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_values = []
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_genetator = None

    def get_data(self, task):
        raise NotImplementedError()

    def get_prompt_generator(self):
        raise NotImplementedError()

    def get_id_column_name(self):
        raise NotImplementedError()

    def process_id_column(self, task, id_column):
        raise NotImplementedError()

    def initialize_task(self, task, description=None):
        if self.get_data(task) is None:
            return

        column_names = list(self.get_data(task).columns)
        # Assume ID column can only appear in first 3 columns
        if len(column_names) >= 3:
            column_names = column_names[:3]
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        if not description:
            description = task.metadata["description"]
        self.prompt_genetator = self.get_prompt_generator()(
            data_description=description,
            column_names=column_names,
            label_column=task.metadata["label_column"],
        )

    def transform(self, task: PredictionTask) -> PredictionTask:
        if self.get_data(task) is None:
            setattr(task, self.get_id_column_name(), None)
            return task

        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        id_column_name = self.get_id_column_name()

        if parser_output[id_column_name] == NO_ID_COLUMN_IDENTIFIED:
            logger.warning(
                "Failed to infer ID column with data descriptions"
                "Retry the inference without data descriptions."
            )
            self.initialize_task(
                task,
                description="Missing data description. Please infer the ID column based on given column names.",
            )
            parser_output = self._chat_and_parse_prompt_output()

        id_column = parser_output[id_column_name]
        id_column = self.process_id_column(task, id_column)
        self.log_value(id_column_name, id_column)
        setattr(task, id_column_name, id_column)
        return task


class TestIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.test_data

    def get_prompt_generator(self):
        return TestIDColumnPromptGenerator

    def get_id_column_name(self):
        return "test_id_column"

    def process_id_column(self, task, id_column):
        if task.output_id_column != NO_ID_COLUMN_IDENTIFIED:
            # if output data has id column but test data does not
            if id_column == NO_ID_COLUMN_IDENTIFIED:
                if task.output_id_column not in task.test_data:
                    id_column = task.output_id_column
                else:
                    id_column = "id_column"
                new_test_data = task.test_data.copy()
                new_test_data[id_column] = task.sample_submission_data[
                    task.output_id_column
                ]
                task.test_data = new_test_data
        return id_column


class TrainIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.train_data

    def get_prompt_generator(self):
        return TrainIDColumnPromptGenerator

    def get_id_column_name(self):
        return "train_id_column"

    def process_id_column(self, task, id_column):
        if id_column != NO_ID_COLUMN_IDENTIFIED:
            new_train_data = task.train_data.copy()
            new_train_data = new_train_data.drop(columns=[id_column])
            task.train_data = new_train_data
            logger.info(f"Dropping ID column {id_column} from training data.")
            task.metadata["dropped_train_id_column"] = True

        return id_column


class OutputIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.sample_submission_data

    def get_prompt_generator(self):
        return OutputIDColumnPromptGenerator

    def get_id_column_name(self):
        return "output_id_column"

    def process_id_column(self, task, id_column):
        return id_column


class EvalMetricInference(TaskInference):
    def initialize_task(self, task):
        problem_type = task.problem_type
        self.metrics = (
            METRICS_DESCRIPTION.keys()
            if problem_type is None
            else METRICS_BY_PROBLEM_TYPE[problem_type]
        )
        self.valid_values = self.metrics
        if problem_type:
            self.fallback_value = METRICS_BY_PROBLEM_TYPE[problem_type][0]
        self.prompt_genetator = EvalMetricPromptGenerator(
            data_description=task.metadata["description"], metrics=self.metrics
        )
