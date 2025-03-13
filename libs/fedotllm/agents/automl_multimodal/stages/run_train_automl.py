import os

import pandas as pd

from typing import Union, List

from sklearn.metrics import f1_score as f1

from pathlib import Path

from fedotllm.settings.config_loader import get_settings
from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState
from fedotllm.data import Dataset

from fedot import Fedot
from golem.core.dag.graph_utils import graph_structure
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedotllm.log import get_logger
logger = get_logger()

def calculate_validation_metric(valid: Union[InputData, MultiModalData], pred: OutputData) -> float:
    """
    Calculates F1 score for predicted data

    :param valid: dataclass with true target
    :param pred: dataclass with model's prediction
    """

    real = valid.target
    predicted = pred.predict

    err = f1(y_true=real,
             y_pred=predicted, average='micro')

    return round(err, 2)

def prepare_multi_modal_data(files_path: str, 
                             num_fields: List[str],
                             text_fields: List[str],
                             target: str,
                             task: Task, 
                             images_size: tuple = (128, 128)) -> MultiModalData:
    """
    Imports data from 3 different sources (table, images and text)

    :param files_path: path to data
    :param task: task to solve
    :param images_size: the requested size in pixels, as a 2-tuple of (width, height)
    :return: MultiModalData object which contains table, text and image data
    """

    path = files_path

    if not os.path.exists(path):
        raise FileNotFoundError(path)


    # import of table data
    data_num = InputData.from_json_files(path, 
                                         fields_to_use=num_fields,
                                         label=target, 
                                         task=task, 
                                         is_multilabel=True, 
                                         shuffle=False)

    class_labels = data_num.target

    img_files_path = f'{files_path}/*.jpeg'
    img_path = img_files_path

    # import of image data
    data_img = InputData.from_image(images=img_path, labels=class_labels, task=task, target_size=images_size)
    # import of text data
    data_text = InputData.from_json_files(path, fields_to_use=text_fields,
                                          label=target, 
                                          task=task,
                                          data_type=DataTypesEnum.text, 
                                          is_multilabel=True, 
                                          shuffle=False)

    data = MultiModalData({
        'data_source_img': data_img,
        'data_source_table': data_num,
        'data_source_text': data_text
    })

    return data

def run_train_automl(state: AutoMLMultimodalAgentState, dataset: Dataset):
    config = state['fedot_config']
    
    task = Task(config.problem.value)
    reflection = state['reflection']
    images_size = (224, 224)

    timeout = config['timeout'],
    
    num_features =  reflection.num_features
    text_features = reflection.text_features
    target = reflection.target

    data = prepare_multi_modal_data(dataset.path, num_features, text_features,
                                    target, task, images_size)
    fit_data, predict_data = train_test_data_setup(data, shuffle=True, split_ratio=0.6)

    logger.info(f"Running AutoML train with timeout of {timeout} mins")

    model = Fedot(problem=config.problem.value,
            timeout=timeout, 
            seed=config.seed,
            cv_folds=config.cv_folds,
            preset=config.preset.value,
            metric=config.metric.value,
            n_jobs=1,
            with_tuning=True,
            show_progress=True)


    pipeline = model.fit(features=fit_data,
                        target=fit_data.target)
    
    logger.info("Pipeline fit")

    save_path = Path(get_settings()['config']['output_dir']) / "pipeline.png"
    pipeline.show(save_path = save_path, font_size_scale = 0.5)

    logger.info("Running evaluate")
    
    
    # TODO: Fix occasional predict crashes. Works perfectly normal outside
    # project, says "'data_source_text'" cannot be used as operation
    # Predict
    prediction = pipeline.predict(predict_data, output_mode='labels')
    err = calculate_validation_metric(predict_data, prediction)
    logger.debug(f"Evaluate result\nF1: {err}")

    # Save submission
    submission_path = Path(get_settings()['config']['output_dir']) / "submission.csv"
    predicted = prediction.predict
    predicted = pd.DataFrame(predicted)
    predicted.to_csv(submission_path)

    state['metrics'] = {'F1 micro': err}
    state['pipeline'] = graph_structure(model.current_pipeline)
        
    # Save the pipeline
    pipeline.save(path=Path(get_settings()['config']['output_dir']) / "pipeline", 
                create_subdir=False, is_datetime_in_path=False)

    return state
