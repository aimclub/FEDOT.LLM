def evaluate_model(model: Fedot, test_features: np.ndarray | pd.DataFrame | pd.Series, test_target: np.ndarray | pd.DataFrame | pd.Series):
    if isinstance(test_features, pd.DataFrame) and isinstance(test_target, (pd.DataFrame, pd.Series)):
        input_data = InputData.from_dataframe(test_features, test_target, task=Task({%problem%}))
    elif isinstance(test_features, np.ndarray) and isinstance(test_target, np.ndarray):
        input_data = InputData.from_numpy(test_features, test_target, task=Task({%problem%}))
    else:
        raise ValueError("Unsupported data types for test_features and test_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(test_features)} and {type(test_target)}")
    y_pred = model.{%predict_method%}
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()