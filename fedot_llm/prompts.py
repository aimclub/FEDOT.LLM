from langchain_core.prompts import ChatPromptTemplate

dataset_name_template = ChatPromptTemplate([
    ('system', 'Define the name of this dataset. Answer only with the name.'),
    ('human', '{dataset_description}')
])

dataset_description_template = ChatPromptTemplate([
    ('system', ('Formulate a short description this dataset.'
                'It should be no longer than a paragraph')),
    ('human', '{dataset_description}')
])

train_split_template = ChatPromptTemplate([
    ('system', 'Define the train split of this dataset. Answer only with the name of the file with train split. Mind the register.'),
    ('human', '{detailed_description}')
])

test_split_template = ChatPromptTemplate([
    ('system', 'Your task is to define the test split of this dataset. Answer only with the name of the file with test split. Mind the register.'),
    ('human', '{detailed_description}')
])

dataset_goal_template = ChatPromptTemplate([
    ('system', ('Your task is to formulate the task associated with this dataset. Answer only with the task description, mention the target column.'
                'It should be short, 1-2 sentences long.')),
    ('human', '{dataset_description}')
])

target_definition_template = ChatPromptTemplate([
    ('system', 'Your task is to return the target column of the dataset. Only answer with a column name'),
    ('human', '{detailed_description}')
])

task_definition_template = ChatPromptTemplate([
    ('system', 'Your task is to define whether the task is regression or classification. Only answer with a task type'),
    ('human', '{detailed_description}')
])


describe_column_template = ChatPromptTemplate([ 
    ('system', "You are helpful AI assistant."
                "User will enter one column from dataset,"
                "and the assistant will make one sentence discription of data in this column."
                "Don't make assumptions about what values to plug into functions. Use column hint."
                "Wrap the output in `json` tags\n{format_instructions}"
                ),
    ('human',"Dataset Title: {dataset_title}\n"
            "Dataset description: {dataset_description}\n"
            "Column name: {column_name}\n"
            "Column hint: {column_hint}\n"
            "Column values:\n"
            "```\n"
            "{column_samples}\n"
            "```")
])

categorical_definition_prompt = '''Your task is to return the list of all option feature columns
                            Only answer with a column name on separate lines'''

categorical_definition_context = '''An option feature column in a dataset:
                            - Can have numeric or string values
                            - Represents state or option, not highly varying quantitative or unique data
                            - Has a low unique value ratio
                            - Not necessaily ordered'''

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

split_description_prompt = '''Your task is to describe the purpose of this dataset split.
It should be short, 1-2 sentences long'''

target_split_definition_prompt = '''Your task is to define the target split of this dataset. Answer only with the name of the file with target split. Mind the register.'''

