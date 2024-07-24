dataset_name_prompt = '''Your task is to define the name of this dataset. Answer only with the name.'''

dataset_description_prompt = '''Your task is to formulate a short description this dataset.
It should be no longer than a paragraph'''

train_split_definition_prompt = '''Your task is to define the train split of this dataset. Answer only with the name of the file with train split. Mind the register.'''

target_split_definition_prompt = '''Your task is to define the target split of this dataset. Answer only with the name of the file with target split. Mind the register.'''

dataset_goal_prompt = '''Your task is to formulate the task associated with this dataset. Answer only with the task description, mention the target column.
It should be short, 1-2 sentences long.'''

split_description_prompt = '''Your task is to describe the purpose of this dataset split.
It should be short, 1-2 sentences long'''

categorical_definition_prompt = '''Your task is to return the list of all option feature columns
                            Only answer with a column name on separate lines'''

categorical_definition_context = '''An option feature column in a dataset:
                            - Can have numeric or string values
                            - Represents state or option, not highly varying quantitative or unique data
                            - Has a low unique value ratio
                            - Not necessaily ordered'''

target_definition_prompt = '''Your task is to return the target column of the dataset
                            Only answer with a column name'''

task_definition_prompt = '''Your task is to define whether the task is regression or classification
                            Only answer with a task type'''