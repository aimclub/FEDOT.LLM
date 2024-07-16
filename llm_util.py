import os
import json
import requests

MODEL_CHAT_ENDPOINT = os.getenv("MODEL_CHAT_ENDPOINT")
if MODEL_CHAT_ENDPOINT is None:
    raise ValueError("MODEL_CHAT_ENDPOINT environment variable is not set")


def generate_dataset_split_description(dataset_description, dataset_columns):
    return f"The {dataset_description} split contains following columns: {dataset_columns}.\n"


def generate_dataset_standard_description(dataset_metadata):

    splits = dataset_metadata["splits"]
    train_split = splits[dataset_metadata["train_split_name"]]
    split_descriptions = dataset_metadata['split_descriptions']

    text_columns = list(train_split.select_dtypes(include=['object']).columns)
    numeric_columns = list(
        train_split.select_dtypes(include=['number']).columns)

    unique_count = train_split.apply(lambda col: col.nunique())
    unique_ratio = train_split.apply(lambda col: col.nunique() / len(col))
    column_types = train_split.apply(
        lambda col: "string" if col.name in text_columns else "numeric")

    column_descriptions = [f"{column_name}: {column_types[column_name]}"
                           f"{100 * unique_ratio[column_name]:.2f}% unique values, examples: {
                               list(train_split[column_name].head(10))} "
                           for column_name in train_split.columns]

    dataset_name = dataset_metadata['name']
    dataset_description = dataset_metadata['description']
    dataset_goal = dataset_metadata['goal']
    dataset_split_names = dataset_metadata['split_names']

    split_description_lines = [generate_dataset_split_description(split_descriptions[split_name], list(
        splits[split_name].columns)) for split_name in dataset_split_names]

    introduction_lines = [
        f"Assume we have a dataset called '{dataset_name}', which describes {
            dataset_description}, and the task is to {dataset_goal}.",
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

    system_prompt = "\n".join(["Return the target column of the dataset. Output: only target column name.",
                               "Q: Predict sales based on the money spent on different platforms for marketing.  colums: TV, Radio, Newspaper, Sales",
                               "A: Sales",
                               "Q: Predict the probability of various defects on steel plates. colums: X_Minimum, Y_Minimum, Pixels_Areas, Faults",
                               "A: Faults"])

    splits = dataset_metadata["splits"]
    columns = ', '.join(
        list(splits[dataset_metadata["train_split_name"]].columns))
    task_prompt = '\n'.join([f"Q: {dataset_metadata['goal']}.",
                            f"columns: {columns}",
                            "A:"])

    return system_prompt, task_prompt


def generate_task_definition_prompts(dataset_metadata):

    task_lines = [
        f"Your task is to define whether the task is regression or classification",
        f"Only answer with a task type"
    ]

    system_prompt = generate_dataset_standard_description(dataset_metadata)
    task_prompt = "\n".join(task_lines)

    return system_prompt, task_prompt


def run_model_multicall(dataset_metadata):
    prompts = {
        "categorical_columns": generate_categorical_definition_prompts(dataset_metadata),
        "target_column": generate_target_definition_prompts(dataset_metadata),
        "task_type": generate_task_definition_prompts(dataset_metadata)
    }

    return invoke_model(prompts)


def invoke_model(prompts):
    data_temp = {'model': "llama3"}
    responses = {}
    for key, value in prompts.items():
        system_prompt, task_prompt = value
        data = data_temp.copy()
        data['messages'] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
        response = requests.post(MODEL_CHAT_ENDPOINT, json=data).json()
        responses[key] = response['choices'][0]['message']['content']
    return responses


def process_model_responses(responses):
    responses["categorical_columns"] = responses["categorical_columns"].split(
        "\n")
    responses["task_type"] = responses["task_type"].lower()
    return responses


def save_model_responses(responses, path):
    with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
        json.dump(responses, json_file)
