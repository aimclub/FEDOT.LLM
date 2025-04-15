from typing import List


def field_parsing_prompt(fields: List[str]) -> str:
    return (
        f"Based on the above information, provide the correct values for the following fields strictly "
        f"in valid JSON format: {', '.join(fields)}.\n\n"
        "Important:\n"
        "1. Return only valid JSON. No extra explanations, text, or comments.\n"
        "2. Ensure that the output can be parsed by a JSON parser directly.\n"
        "3. Do not include any non-JSON text or formatting outside the JSON object."
        '4. An example is \{"<provided_field>": "<correct_value_for_the_field>"\}'
    )
