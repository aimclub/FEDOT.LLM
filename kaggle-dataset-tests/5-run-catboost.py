import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

train = pd.read_csv("../datasets-offline/playground-series-s4e7/train.csv")
test = pd.read_csv("../datasets-offline/playground-series-s4e7/test.csv")
sub = pd.read_csv("../datasets-offline/playground-series-s4e7/sample_submission.csv")

train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


def preprocess(df):
    df["Vehicle_Age"] = df["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2})
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map({"No": 0, "Yes": 1})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df


train, test = [preprocess(df) for df in [train, test]]

print(train.size)
print(test.size)

auto_model = Fedot(
    problem="classification",
    metric=["roc_auc"],
    preset="best_quality",
    with_tuning=True,
    timeout=5,
    cv_folds=10,
    seed=42,
    n_jobs=1,
    logging_level=10,
    initial_assumption=PipelineBuilder()
    .add_node(
        "scaling",
    )
    .add_node(
        "catboost",
        params={"use_eval_set": True, "use_best_model": True, "iterations": 10000, "n_jobs": -1},
    )
    .build(),
    use_pipelines_cache=False,
    use_auto_preprocessing=False,
)

auto_model.fit(features=train, target="Response")

prediction = auto_model.predict_proba(features=test)

sub.Response = prediction.ravel()
sub.to_csv("submission.csv", index=False)

print(auto_model.return_report().head(10))

auto_model.current_pipeline.show()

auto_model.current_pipeline.save(path="./saved_pipelines", create_subdir=True, is_datetime_in_path=True)
