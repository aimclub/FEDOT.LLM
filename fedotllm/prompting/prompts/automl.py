from typing import List

import fedotllm.prompting.prompts.utils as prompt_utils
from fedotllm.constants import (
    METRICS_DESCRIPTION,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    PROBLEM_TYPES,
    TASK_TYPES,
)


def task_type_prompt(data_description: str, fields: List[str]) -> str:
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Your Task
Based on the information provided, identify the correct task_type to be used 
from among these KEYS: {", ".join(TASK_TYPES)}

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


def data_file_name_prompt(data_description: str, filenames: str, fields: List[str]):
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Available Data File And Columns in The File
{filenames}

# Your Task
Based on the data description, what are the training, test, and output data?
The output file may contain keywords such as benchmark, submission, or output.
Please return the full path of the data files as provided, and response with the value {NO_FILE_IDENTIFIED} if there's no such File.

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


def label_column_prompt(
    data_description: str, column_names: List[str], fields: List[str]
):
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Available Columns
{column_names}

# Your Task
Based on the data description, which one of these columns is likely to be the label column?

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


def problem_type_prompt(data_description: str, fields: List[str]):
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Your Task
Based on the information provided, identify the correct problem_type to be used 
from among these KEYS: {", ".join(PROBLEM_TYPES)}

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


def id_column_prompt(
    data_description: str, column_names: List[str], label_column: str, fields: List[str]
):
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Available Columns
{column_names}

# Your Task
Based on the data description, which one of these columns is likely to be the Id column?
If no reasonable Id column is preset, for example if all the columns appear to be similarly named feature columns, 
response with the value {NO_ID_COLUMN_IDENTIFIED}
ID columns can't be {label_column}

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


def evaluation_metrics_prompt(
    data_description: str, metrics: List[str], fields: List[str]
):
    return f"""
You are an expert assistant that parses information about data science tasks.

# Data Description
{data_description}

# Your Task
Based on the information provided, identify the correct evaluation metric to be used from among these KEYS:
{", ".join(metrics)}
The descriptions of these metrics are:
{", ".join([METRICS_DESCRIPTION[metric] for metric in metrics])}
respectively.
If the exact metric is not in the list provided, then choose the metric that you think best approximates the one in the task description.
Only respond with the exact names of the metrics mentioned in KEYS. Do not respond with the metric descriptions.

# Output
{prompt_utils.field_parsing_prompt(fields)}
"""


# ## Available Tools:
# Each tool is described in JSON format. When you call a tool, **import the tool from its path first.**
# {tool_schemas}


def code_generation_prompt(
    user_instruction: str,
    dataset_path: str,
    files: str,
    skeleton: str,
) -> str:
    return f"""
As a data scientist, you need to help user to achieve their goal step by step in a Python code.
# User Requirement
{user_instruction}

# Tool Info
## Capabilities
- You can utilize pre-defined tools in any code lines from 'Available Tools' in the form of Python class or function.
- You can freely combine the use of any other public packages, like sklearn, numpy, pandas, etc..

# About Dataset

[Path to Dataset]
{dataset_path}
[Files]
{files}
[solution.py] 
```python
{skeleton} 
```
Start the python code with "```python".
Please ensure the completeness of the code so that it can be run without additional modifications.

# Constraints

1. You are obliged **NOT DELETE ANY COMMENTS**.
2. You can't change the code between ### comment ### code ### comment ###. This code will be regenerated.
3. You **can't use methods and attributes of Fedot framemork classes (Fedot, Pipeline)**, except those **already used** in the code or **provided in the comments**. 
4. You are obliged to write code in the 'USER CODE' sections. The rest of the code will be regenerated when the project is restarted. 

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
```python
your code
```
"""


def fix_solution_prompt(
    user_instruction: str,
    dataset_path: str,
    files: str,
    code_recent_solution: str,
    stdout: str,
    stderr: str,
) -> str:
    return f"""
You're a machine learning expert.
You are given a machine learning problem.
# About Dataset
[Task]
{user_instruction}
[Path to Dataset]
{dataset_path}
[Files]
{files}

A previous Python solution code was generated for the problem:
```python
{code_recent_solution}
```
Output:
<stdout>
{stdout}
</stdout>
Problem trace:
<stderr>
{stderr}
</stderr>

You need to generate a fixed solution code.
- **Concentrate on solving the problem that caused the error**, don't fix others 
- **Write the whole code**
Answer:
```python
"""


def generate_configuration_prompt(reflection: str, dataset_description: str) -> str:
    return f"""
You're a machine learning expert.
You are given a machine learning problem.
<problem_reflection>
{reflection}
</problem_reflection>

<dataset-description>
{dataset_description}
</dataset-description>
Your goal is to define the optimal parameters for an automatic machine learning model fitting framework.
Make sure to fully address the problem goals, rules and constraints.
Use default values if not specified.
"""


def problem_reflection_prompt(
    user_description: str, data_files_and_content: str
) -> str:
    return f"""
Please conduct a comprehensive analysis of the competition, focusing on the following aspects:
1. Competition Overview: Understand the background and context of the topic.
2. Files: Analyze each provided file, detailing its purpose and how it should be used in the competition.
3. Problem Definition: Clarify the problem's definition and requirements.
4. Data Information: Gather detailed information about the data, including its structure and contents.
    4.1 Data type:
        4.1.1. ID type: features that are unique identifiers for each data point, which will NOT be used in the model training.
        4.1.2. Numerical type: features that are numerical values.
        4.1.3. Categorical type: features that are categorical values.
        4.1.4 Datetime type: features that are datetime values.
    4.2 Detailed data description
5. Target Variable: Identify the target variable that needs to be predicted or optimized, which is provided in the training set but not in the test set.
6. Evaluation Metrics: Determine the evaluation metrics that will be used to assess the submissions.
7. Submission Format: Understand the required format for the final submission.
8. Other Key Aspects: Highlight any other important aspects that could influence the approach to the competition.
Ensure that the analysis is thorough, with a strong emphasis on :
1. Understanding the purpose and usage of each file provided.
2. Figuring out the target variable and evaluation metrics.
3. Classification of the features.

# User Description
{user_description}

# Available Data File And Content in The File
{data_files_and_content}
"""


def reporter_prompt(description: str, metrics: str, pipeline: str, code: str) -> str:
    return f"""
You are an expert in ML. You always write clearly and concisely.
You've created an ML model to solve the problem:
```
{description}
```
Send a messenger message with a report of your findings.
Characteristics of the resulting ML model:
Metrics: {metrics}
Pipeline: 
```
{pipeline}
```
Code:
```python
{code}
```
1. Be concise. 
2. Structure your text using bullet points where appropriate.
3. Use Markdown formatting.
4. Use tables.
5. Don't talk about empty values.
6. Style is Twitter post.
7. Do not suggest next steps.
"""
