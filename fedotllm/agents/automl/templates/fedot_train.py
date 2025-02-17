def train_model(train_features: np.ndarray, train_target: np.ndarray):
    input_data = InputData.from_numpy(train_features, train_target, task=Task({%problem%}))
    model = Fedot(problem={%problem%}.value,
            timeout={%timeout%},
            seed=42,
            cv_folds={%cv_folds%},
            preset={%preset%},
            metric={%metric%},
            n_jobs=1,
            with_tuning=True,
            show_progress=True)

    model.fit(features=input_data) # this is the training step, after this step variable ‘model‘ will be a trained model

    # Save the pipeline
    pipeline = model.current_pipeline
    pipeline.save(path=PIPELINE_PATH, create_subdir=False, is_datetime_in_path=False)

    return model