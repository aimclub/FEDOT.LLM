from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

dataset_name_template = ChatPromptTemplate([
    ('system', 'Define a concisethe name of this dataset. Answer only with the name.'),
    ('human', '{big_description}')
])
"""INPUT: 
- big_description -- user input big description of dataset"""

dataset_description_template = ChatPromptTemplate([
    ('system', ('Formulate a short description this dataset.'
                'It should be no longer than a paragraph. ')),
    ('human', '{big_description}'),
    ('ai', 'Here is a short description of the dataset:\n\n')
])
"""INPUT: 
- big_description -- user input big description of dataset"""

dataset_goal_template = ChatPromptTemplate([
    ('system', ('Formulate the goal associated with this dataset. Write a concisethe goal description.'
                'It should be 1 sentences long.')),
    ('human', '{big_description}'),
    ('ai', 'The goal is\n')
])
"""INPUT: 
- big_description -- user input big description of dataset"""

train_split_template = ChatPromptTemplate([
    ('system', 'Define the train split of this dataset. Answer only with the name of the file with train split. Mind the register.'),
    ('human', '{detailed_description}')
])
"""INPUT:
- detailed_description: property of the dataset object"""

test_split_template = ChatPromptTemplate([
    ('system', 'Your task is to define the test split of this dataset. Answer only with the name of the file with test split. Mind the register.'),
    ('human', '{detailed_description}')
])
"""INPUT:
- detailed_description: property of the dataset object"""


target_definition_template = ChatPromptTemplate([
    ('system', 'Your task is to return the target column of the dataset. Only answer with a column name.'),
    ('human', '{detailed_description}'),
    ('ai', 'Target column:\n')
])
"""INPUT:
- detailed_description: property of the dataset object"""

task_definition_template = ChatPromptTemplate([
    ('system', 'Your task is to define whether the task is regression or classification. Only answer with a task type'),
    ('human', '{detailed_description}')
])
"""INPUT:
- detailed_description: property of the dataset object"""


describe_column_template = ChatPromptTemplate([
    ('system', "You are helpful AI assistant."
     "User will enter one column from dataset,"
     "and the assistant will make one sentence discription of data in this column."
     "Don't make assumptions about what values to plug into functions."
     "Wrap the output in `json` tags\n{format_instructions}"
     ),
    ('human', "Dataset Title: {dataset_title}\n"
     "Dataset description: {dataset_description}\n"
     "Column name: {column_name}\n"
     "Column values:\n"
     "```\n"
     "{column_samples}\n"
     "```")
])
"""INPUT: format_instructions, dataset_title, dataset_description, column_name, column_hint, column_samples"""

categorical_examples = [
    {"input": "Column name: Gender\nColumn data: male, female, male, male\nColumn unique ratio: 0.0",
     "output": 'Based on the column description and data, I can conclude that:\n\n1. Unique ratio is low. 2. Data is consists of geneder labels, hence it is nominal data hence categorical data.'
     'The conclusion is a categorical feature, because ratio is low -> few unique values, and consists of geneder label -> nominal data .\n\n'
     '```json\n {"name": "Gender", "column_type": "categorical"}\n```'},
    {"input": 'Column name: Age\nColumn data: 25, 30, 35, 40\nColumn unique ratio: 0.8',
     "output": 'Based on the column description and data, I can conclude that:\n\n1. Unique ratio is hight. 2. We count years lived by people, we can add or subtract years.'
     'Hence it is discrete data hence numerical feature. The conclusion is a numerical feature, because ratio is high -> many unique values, and we can add or subtract years -> discrete data.\n\n'
     '\n\n```json\n {"name": "Age", "column_type": "numerical"}\n```'},
    {"input": 'Column name: Name\nColumn data: Alex, Stanislav, Nikolay, Elena\nColumn unique ratio: 1.0',
     "output": 'Based on the column description and data, I can conclude that:\n\n1. Unique ratio says to us that all values unique. 2. We can add new names to the dataset.'
     'Hence it is numerical data. The conclusion is a numerical feature, because ratio is 1 -> all values unique, and we can add new names -> discrete data.\n\n'
     '\n\n```json\n {"name": "Name", "column_type": "numerical"}\n```'},


]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
categorica_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=categorical_examples,
)

categorical_fix_examples = [
    {"input": "Completion:```json\n {\"name\": \"Region_Code\", \"column_type\": \"Discrete\"}\n```",
     "output": "```json\n {\"name\": \"Region_Code\", \"column_type\": \"numerical\"}"},
    {"input": "Completion:```json\n {\"name\": \"Gender\", \"column_type\": \"Categorical\"}\n```",
        "output": "```json\n {\"name\": \"Gender\", \"column_type\": \"categorical\"}"},
    {"input": "Completion:```json\n {\"name\": \"Age\", \"column_type\": \"Ddiscretee\"}\n```",
        "output": "```json\n {\"name\": \"Age\", \"column_type\": \"numerical\"}"},
]

categorica_fix_example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=categorical_fix_examples,
)

categorical_fix_template = ChatPromptTemplate([
    ('system', 'You need to fix the output. The output should be formatted as a JSON instance that conforms to the JSON schema below.'
     '```json\n {{"name": "column_name", "column_type": "categorical|numerical"}}\n'
     '```'),
    categorica_fix_example_prompt,
])

categorical_template = ChatPromptTemplate([
    ('system', 'Be specific and respond in the tone of a McKinsey consultant. '
     'User will enter one column from dataset, '
     'and the assistant will define whether the column consists of categorical data or numerical data.'
     "Wrap the output in `json` tags\n{format_instructions}"
     'CONTEXT:'
     'Categorical data refers to a data type that can be stored and identified based on the names or labels given to them. A process called matching is done to draw out the similarities or relations between the data and then grouped accordingly.'
     'The data collected in the categorical form is also known as qualitative data. Each dataset can be grouped and labeled depending on their matching qualities under only one category. This makes the categories mutually exclusive. '
     'Types of categorical data: 1. Nominal data\n'
     'This is also called naming data. This type names or labels the data, and its characteristics are similar to a noun. Example: person’s name, gender, school name.'
     '2. Ordinal data\n'
     'This includes data or elements of data that are ranked, ordered, or used on a rating scale. You can count and order ordinal data, but it doesn’t allow you to measure it.'
     'Example: seminar attendants are asked to rate their seminar experience on a scale of 1-5. Against each number, there will be options that will rate their satisfaction, such as “very good, good, average, bad, and very bad.”'
     'Numerical data refers to the data that is in the form of numbers, and not in any language or descriptive form. Often referred to as quantitative data, numerical data is collected in number form and stands different from any form of number data type due to its ability to be statistically and arithmetically calculated.'
     'Types of numerical data'
     '1. Discrete data'
     'Discrete data is used to represent countable items. It can take both numerical and categorical forms and group them into a list. This list can be finite or infinite, too.'
     'Discrete data takes countable numbers like 1, 2, 3, 4, 5, and so on. In the case of infinity, these numbers will keep going on. '
     'Example: counting sugar cubes from a jar is finite countable. But counting sugar cubes from all over the world is infinitely countable.'
     '2. Continuous data'
     'As the name says, this form has data in the form of intervals. Or simply said ranges. Continuous numerical data represent measurements; their intervals fall on a number line. Hence, it doesn’t involve taking counts of the items. '
     'Interval data – interval data type refers to data that can be measured only along a scale at equal distances from each other. The numerical values in this data type can only undergo add and subtract operations. For example, body temperature can be measured in degrees Celsius and degrees Fahrenheit, and neither of them can be 0.'
     'Ratio data – unlike interval data, ratio data has zero points. Similar to interval data, zero points are the only difference they have. For example, in the body temperature, the zero point temperature can be measured in Kelvin.'
    ),
    categorica_few_shot_prompt,
    ('user',    'Dataset Title: {dataset_title} '
                'Dataset description: {dataset_description} '
                'Column name: {column_name} '
                'Column description: {column_description} '
                'Column unique ratio: {column_ratio} (number of unique values divided by the number of total values) '
                'Column data:\n'
                '```\n'
                '{column_samples}\n'
                '```'
    )
])
"""INPUT format_instructions, dataset_title, dataset_description, column_name, column_description, column_ratio, column_samples"""

analyze_predictions = ChatPromptTemplate([
    ('user', 'You are an expert in ML. You always write clearly and concisely. '
               'Describe pipeline of model that you build: '
               '{parameters}'
               'Tell about the model metrics: {metrics}'
               'Explain what each metric means.'
               "Don't talk about empty values. Don't talk about things that aren't in the configuration dictionary."
               "Don't make assumptions about type of scaling."
               'Use Markdown formatting.'
               'Start with: "Here is the pipeline of the model I built:"'
               '<output>'
               '# Model Pipeline\n'
               'The pipeline consists of'
               '<stages_description>'
               '</stages_description>'
               '# Model Metrics:'
               '| Metric | Value |'
                '| --- | --- |'
                '<metrics_table/>'
                'These metrics indicate that'
                '<metrics_description/>'
                '</output>'
               )
    
])
