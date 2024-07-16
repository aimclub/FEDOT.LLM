import logging

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def run_example(train_df, test_df, dataset_metadata,
                cv_folds=10, metric=['roc_auc', 'accuracy'], n_jobs=-1,
                visualise: bool = False, with_tuning: bool = True,
                timeout: float = 2., preset: str = 'auto'):

    problem = dataset_metadata["task_type"].lower()

    composer_params = {'history_dir': 'custom_history_dir', 'preset': preset}
    auto_model = Fedot(problem=problem, seed=42, timeout=timeout, cv_folds=cv_folds,
                       metric=metric, logging_level=logging.FATAL,
                       with_tuning=with_tuning, **composer_params)

    auto_model.fit(features=train_df,
                   target=dataset_metadata['target_column'].capitalize())
    prediction = auto_model.predict(features=test_df)

    if visualise:
        auto_model.history.save('saved_regression_history.json')
        auto_model.plot_prediction()

    print(auto_model.get_metrics())
    return prediction
