def automl_predict(model, features: pd.DataFrame | pd.Series) -> np.ndarray:
    input_data = InputData.from_numpy(features.to_numpy(), None, task=Task({%problem%}))
    predictions = model.{%predict_method%}
    print(f"Predictions shape: {predictions.shape}")
    return predictions
    