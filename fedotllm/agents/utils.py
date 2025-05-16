import json
import re
from typing import Any, Dict

import json_repair
from jinja2 import Environment, StrictUndefined

from fedotllm.log import logger


def jinja_render(template: str, *args, **kwargs):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(template).render(*args, **kwargs)


def render(prompt, *args, **kwargs):
    system = prompt.get("system", None)
    if system:
        system = jinja_render(system, *args, **kwargs)
    user = jinja_render(prompt.user, *args, **kwargs)

    temperature = prompt.get("temperature", 0.2)
    frequency_penalty = prompt.get("frequency_penalty", 0.0)

    return user, system, temperature, frequency_penalty


# if ```python on response, or ``` on response, or whole response is code, return code
def extract_code(response: str) -> str:
    """Extract code content from text that may contain code blocks.

    Args:
        response: Input text that might contain code blocks
    Returns:
        Extracted code content or original text if no code blocks found
    """
    response = response.strip()
    code_match = re.search(
        r"```(?:\w+)?\s*(.*?)```",
        response,
        re.DOTALL,
    )
    return code_match.group(1).strip() if code_match else response


def parse_json(raw_reply: str) -> Dict[str, Any] | None:
    def try_json_loads(data: str) -> Dict[str, Any]:
        try:
            return json_repair.repair_json(
                data, ensure_ascii=False, return_objects=True
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return None

    raw_reply = raw_reply.strip()
    # Case 1: Check if the JSON is enclosed in triple backticks
    json_match = re.search(r"\{.*\}|```(?:json)?\s*(.*?)```", raw_reply, re.DOTALL)
    if json_match:
        if json_match.group(1):
            reply_str = json_match.group(1).strip()
        else:
            reply_str = json_match.group(0).strip()
        reply = try_json_loads(reply_str)
        if reply is not None:
            return reply

    # Case 2: Assume the entire string is a JSON object
    return try_json_loads(raw_reply)
