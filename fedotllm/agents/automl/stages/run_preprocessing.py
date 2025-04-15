from logging import getLogger
from pathlib import Path
from typing import List, Type

from langgraph.types import Command
from omegaconf import DictConfig

from fedotllm.agents.automl.stages.preprocessing import (
    DataFileNameInference,
    EvalMetricInference,
    LabelColumnInference,
    OutputIDColumnInference,
    ProblemTypeInference,
    TaskInference,
    TaskTypeInference,
    TestIDColumnInference,
    TrainIDColumnInference,
)
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm import LiteLLMModel
from fedotllm.predictor.task import PredictionTask
from fedotllm.utils.io import append_yaml

logger = getLogger(__name__)


class SimpleObservationAgent:
    def __init__(
        self, config: DictConfig, task_path: Path, llm: LiteLLMModel, description: str
    ) -> None:
        self.config = config
        self.llm = llm
        self.task = PredictionTask.from_path(task_path, description)

    def _run_task_inference_preprocessors(
        self,
        task_inference_preprocessors: List[Type[TaskInference]],
        task: PredictionTask,
    ):
        for preprocessor_class in task_inference_preprocessors:
            preprocessor = preprocessor_class(llm=self.llm)
            try:
                task = preprocessor.transform(task)
            except Exception as e:
                logger.error(f"Task inference preprocessing: {preprocessor_class}", e)
                raise e

    def inference_task(self) -> PredictionTask:
        logger.info("Task understanding starts...")
        task_inference_preprocessors: List[Type[TaskInference]] = [
            DataFileNameInference,
            LabelColumnInference,
            TaskTypeInference,
            ProblemTypeInference,
            OutputIDColumnInference,
            TrainIDColumnInference,
            TestIDColumnInference,
            EvalMetricInference,
        ]

        self._run_task_inference_preprocessors(task_inference_preprocessors, self.task)

        logger.info("Task understanding complete!")
        return self.task


def run_preprocessing(state: AutoMLAgentState, config: DictConfig, llm: LiteLLMModel):
    inference_agent = SimpleObservationAgent(
        config, state["task_path"], llm, state["reflection"]
    )
    task = inference_agent.inference_task()
    append_yaml(task.to_dict(), state["workspace"] / "exploration.yaml")
    return Command(update={"prediction_task": task})
