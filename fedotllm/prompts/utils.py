from pydantic import BaseModel
from typing import TypeVar, Type

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
