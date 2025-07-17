def automl_predict(model: Fedot, features: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(features, (pd.DataFrame, pd.Series)):
        features = features.to_numpy()
    input_data = InputData.from_numpy(features, None, task=Task({%problem%}))
    predictions = model.{%predict_method%}
    print(f"Predictions shape: {predictions.shape}")
    return predictions