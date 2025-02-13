def automl_predict(model, features: np.ndarray) -> np.ndarray:
    input_data = InputData.from_numpy(features, None, task=Task({%problem%}))
    return model.{%predict_method%}
    