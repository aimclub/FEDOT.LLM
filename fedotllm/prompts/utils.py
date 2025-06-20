from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def structured_response(response_model: Type[T]):
    return f"""
The output must be a JSON object equivalent to type ${response_model.__name__}, according to the following Pydantic definitions:
=====
```
{response_model.model_json_schema()}
```
=====
Important:
1. Return only valid JSON. No extra explanations, text, or comments.
2. Ensure that the output can be parsed by a JSON parser directly.
3. Do not include any non-JSON text or formatting outside the JSON object.
4. An example is \{{"<object_field>": "<correct_value_for_the_field>"\}}
"""


def ai_assert_prompt(var1, var2, condition: str):
    return f"""
You are an intelligent assertion function to evaluate conditions between two variables.
The variables are:
1. var1: {var1}
2. var2: {var2}
condition: {condition}
=====
Important:
1. Return only true or false. No extra explanations, text, or comments
2. Ensure that the output can be parsed by a regex pattern: ^(true|false)$
3. Do not include any text or formatting outside the true/false value
4. An example is true or false
"""
