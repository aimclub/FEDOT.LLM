def train_model(train_features: np.ndarray | pd.DataFrame, train_target: np.ndarray | pd.DataFrame | pd.Series):
    if isinstance(train_features, pd.DataFrame) and isinstance(train_target, (pd.DataFrame, pd.Series)):
        input_data = InputData.from_dataframe(train_features, train_target, task=Task({%problem%}))
    elif isinstance(train_features, np.ndarray) and isinstance(train_target, np.ndarray):
        input_data = InputData.from_numpy(train_features, train_target, task=Task({%problem%}))
    else:
        raise ValueError("Unsupported data types for train_features and train_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(train_features)} and {type(train_target)}")
        
    model = Fedot(problem={%problem%}.value,
            timeout={%timeout%},
            seed=42,
            cv_folds={%cv_folds%},
            preset={%preset%},
            metric={%metric%},
            n_jobs=1,
            with_tuning=True,
            show_progress=True)

    try:
        model.fit(features=input_data) # this is the training step, after this step variable 'model' will be a trained model
    except Exception as e:
        raise RuntimeError(
            f"Model training failed. Please check your data preprocessing carefully. "
            f"Common issues include: missing values, incorrect data types, feature scaling problems, "
            f"or incompatible target variable format. Original error: {str(e)}"
        ) from e

    # Save the pipeline
    pipeline = model.current_pipeline
    pipeline.save(path=PIPELINE_PATH, create_subdir=False, is_datetime_in_path=False)

    return model