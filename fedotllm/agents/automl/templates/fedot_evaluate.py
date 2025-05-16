def evaluate_model(model, test_features: pd.DataFrame | pd.Series, test_target: pd.DataFrame | pd.Series):
    input_data = InputData.from_dataframe(test_features, test_target, task=Task({%problem%}))
    y_pred = model.{%predict_method%}
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
