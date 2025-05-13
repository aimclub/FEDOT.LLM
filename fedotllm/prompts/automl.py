def code_generation_prompt(
    user_instruction: str, dataset_path: str, files: str, skeleton: str
) -> str:
    return f"""
You are a helpful intelligent assistant. Now please help solve the following machine learning task.
# About Dataset
[Task]
{user_instruction}
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

1. You are obliged **NOT DELETE ANY COMMENTS**.
2. You can't change the code between ### comment ### code ### comment ###. This code will be regenerated.
3. You **can't use methods and attributes of Fedot framemork classes (Fedot, Pipeline)**, except those **already used** in the code or **provided in the comments**. 
4. You are obliged to write code in the 'USER CODE' sections. The rest of the code will be regenerated when the project is restarted. 
```python
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
temperature = 0.2
system = ""
user = '''
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


def problem_reflection_prompt(description: str, dataset_description: str) -> str:
    return f"""
You're a machine learning expert.
You are given a machine learning problem.
<problem_description>
{description}
</problem_description>

<dataset-description>
{dataset_description}
</dataset-description>

Make sure to fully address the problem goals, rules and constraints.
Define modelling task type clear on objective description and sample submission if description contains it.
Your goal is to define the optimal arguments for given function.
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
