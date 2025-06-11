def automl_predict(model, features: np.ndarray) -> np.ndarray:
    input_data = InputData.from_numpy(features, None, task=Task({%problem%}))
    predictions = model.{%predict_method%}
    print(f"Predictions shape: {predictions.shape}")
    return predictions