categorical_definition_prompt = '''Your task is to return the list of all option feature columns
                            Only answer with a column name on separate lines'''

categorical_definition_context = ''' A option feature column in a dataset:
                            - Can have numeric or string values
                            - Represents state or option, not highly varying quantitative or unique data
                            - Has a low unique value ratio
                            - Not necessaily ordered'''

target_definition_prompt = ''' Your task is to return the target column of the dataset
                            Only answer with a column name'''

task_definition_prompt = ''' Your task is to define whether the task is regression or classification
                            Only answer with a task type'''