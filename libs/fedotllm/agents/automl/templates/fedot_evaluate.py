def evaluate_model(model, test_features: np.ndarray, test_target: np.ndarray):
    input_data = InputData.from_numpy(test_features, test_target, task=Task({%problem%}))
    y_pred = model.{%predict_method%}
    return model.get_metrics()