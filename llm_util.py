import os
import json

def generate_dataset_split_description(dataset_description, dataset_columns):
    return f"The {dataset_description} split contains following columns: {dataset_columns}.\n"

def generate_dataset_standard_description(dataset_metadata):
    
    splits = dataset_metadata["splits"]
    train_split = splits[dataset_metadata["train_split_name"]]
    split_descriptions = dataset_metadata['split_descriptions']
    
    text_columns = list(train_split.select_dtypes(include=['object']).columns)
    numeric_columns = list(train_split.select_dtypes(include=['number']).columns)
    
    unique_count = train_split.apply(lambda col: col.nunique())
    unique_ratio = train_split.apply(lambda col: col.nunique() / len(col))
    column_types = train_split.apply(lambda col: "string" if col.name in text_columns else "numeric")
    
    column_descriptions = [f"{column_name}: {column_types[column_name]}"
                           f"{100 * unique_ratio[column_name]:.2f}% unique values, examples: {list(train_split[column_name].head(10)) } "
                           for column_name in train_split.columns]

    dataset_name = dataset_metadata['name']
    dataset_description = dataset_metadata['description']
    dataset_goal = dataset_metadata['goal']
    dataset_split_names = dataset_metadata['split_names']

    split_description_lines = [generate_dataset_split_description(split_descriptions[split_name], list(splits[split_name].columns)) for split_name in dataset_split_names]

    introduction_lines = [
        f"Assume we have a dataset called '{dataset_name}', which describes {dataset_description}, and the task is to {dataset_goal}.",
        f""
    ] + split_description_lines + [
        f"Below is the type (numeric or string), unique value count and ratio for each column, and few examples of values:",
        f""
    ] + column_descriptions + [
        f"",
    ]

    return "\n".join(introduction_lines)

def generate_categorical_definition_prompts(dataset_metadata):

    categorical_variable_info = [
       f"A option feature column in a dataset:",
       f"- Can have numeric or string values",
       f"- Represents state or option, not highly varying quantitative or unique data",
       f"- Has a low unique value ratio",
       f"- Not necessaily ordered",
    ]
    
    task_lines = categorical_variable_info + [
        f"Your task is to return the list of all option feature columns.",
        f"Only answer with a column name on separate lines"
    ]

    system_prompt = generate_dataset_standard_description(dataset_metadata)
    task_prompt = "\n".join(task_lines)

    return system_prompt, task_prompt

def generate_target_definition_prompts(dataset_metadata):
    
    task_lines = [
        f"Your task is to return the target column of the dataset",
        f"Only answer with a column name"
    ]
    
    system_prompt = generate_dataset_standard_description(dataset_metadata)
    task_prompt = "\n".join(task_lines)

    return system_prompt, task_prompt

def generate_task_definition_prompts(dataset_metadata):
    
    task_lines = [
        f"Your task is to define whether the task is regression or classification",
        f"Only answer with a task type"
    ]

    system_prompt = generate_dataset_standard_description(dataset_metadata)
    task_prompt = "\n".join(task_lines)

    return system_prompt, task_prompt

def run_model_multicall(model, tokenizer, generation_config, dataset_metadata):
    prompts = {
        "categorical_columns": generate_categorical_definition_prompts(dataset_metadata),
        "target_column": generate_target_definition_prompts(dataset_metadata),
        "task_type": generate_task_definition_prompts(dataset_metadata)
    }
    
    responses = {}
    for task in prompts:
        system_prompt, task_prompt = prompts[task]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            input_ids,
            **generation_config
        )
        response = outputs[0][input_ids.shape[-1]:]
        responses[task] = tokenizer.decode(response, skip_special_tokens=True)

    return responses

def process_model_responses(responses):
    responses["categorical_columns"] = responses["categorical_columns"].split("\n")
    responses["task_type"] = responses["task_type"].lower()
    return responses

def save_model_responses(responses, path):
    with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
        json.dump(responses, json_file)