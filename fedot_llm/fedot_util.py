import logging

from fedot import Fedot

def run_example(train_df, test_df, problem, target,
                cv_folds = 10, metric = ['roc_auc', 'accuracy'], n_jobs = -1,
                visualise: bool = False, with_tuning: bool = True,
                timeout: float = 2., preset: str = 'auto'):

    problem = problem.lower() 

    composer_params = {'history_dir': 'custom_history_dir', 'preset': preset}
    auto_model = Fedot(problem=problem, seed=42, timeout=timeout, cv_folds=cv_folds,
                       metric=metric, logging_level=logging.FATAL,
                       with_tuning=with_tuning, **composer_params)

    auto_model.fit(features=train_df, target=target)
    prediction = auto_model.predict(features=test_df)

    if visualise:
        auto_model.history.save('saved_regression_history.json')
        auto_model.plot_prediction()

    print(auto_model.get_metrics())
    return prediction
