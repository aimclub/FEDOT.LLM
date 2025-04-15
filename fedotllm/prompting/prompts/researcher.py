def generate_prompt(documents: str, question: str) -> str:
    return f"""
You are DocBot, a helpful assistant that is an expert at helping users with the documentation. \n
Here is the relevant documentation: \n
<documentation>
{documents}
</documentation>
If you don't know the answer, just say that you don't know. Keep the answer concise. \n
When a user asks a question, perform the following tasks:
1. Find the quotes from the documentation that are the most relevant to answering the question. These quotes can be quite long if necessary (even multiple paragraphs). You may need to use many quotes to answer a single question, including code snippits and other examples.
2. Assign numbers to these quotes in the order they were found. Each page of the documentation should only be assigned a number once.
3. Based on the document and quotes, answer the question. Directly quote the documentation when possible, including examples. When relevant, code examples are preferred.
4. When answering the question provide citations references in square brackets containing the number generated in step 2 (the number the citation was found)
5. Structure the output
Example output:
{"citations": [
            {"page_title": "FEDOT 0.7.4 documentation",
                "url": "https://fedot.readthedocs.io/en/latest",
                "number": 1,
                "relevant_passages": [
                        "This example explains how to solve regression task using Fedot.",
                    ]
            }
        ],
    "answer": "The answer to the question."
}
# Question: {question}
"""


def is_grounded_prompt(documents: str, generation: str) -> str:
    return f"""
You are a grader assessing whether an answer is grounded in / supported by a set of facts.
Here are the facts: 
-------
{documents}
-------
Here is the answer: 
{generation}
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""


def is_useful_prompt(generation: str, question: str) -> str:
    return f"""
You are a grader assessing whether an answer is useful to resolve a question.
Here is the answer:
\n ------- \n
{generation}
\n ------- \n
Here is the question: 
{question}
Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""


def retrieve_grader_prompt(document: str, question: str) -> str:
    return f"""
You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document:
\n ------- \n
{document}
\n ------- \n
Here is the user question:
\n ------- \n
{question}
\n ------- \n
If the document contains keywords related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""


def rewrite_question_prompt(question: str) -> str:
    return f"""
You a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the initial and formulate an improved question.
Here is the initial question: 
\n ------- \n
{question}
\n ------- \n
Provide improved question as a JSON with a single key
'question' and no preamble or explanation.
"""
