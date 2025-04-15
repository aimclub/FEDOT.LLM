import difflib
import json
import logging
import re
from typing import Any, Dict, Iterable, Optional

import json_repair
from jinja2 import Environment, StrictUndefined

from fedotllm.exceptions import OutputParserException

logger = logging.getLogger(__name__)


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


def extract_code(content: str) -> Optional[str]:
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        if match.group(1):
            print(f"Group 1: {match.group(1).strip()}")
            return match.group(1).strip()
    match = re.search(r"^\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_json(raw_reply: str) -> Optional[Dict[str, Any]]:
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


def check_json_values(
    parsed_json: Dict,
    valid_values: Optional[Iterable[str]],
    fallback_value: Optional[str],
):
    if valid_values is not None:
        for key, parsed_value in parsed_json.items():
            # Currently only support single parsed value
            if isinstance(parsed_value, list) and len(parsed_value) == 1:
                parsed_value = parsed_value[0]
            if isinstance(parsed_value, str):
                close_matches = difflib.get_close_matches(parsed_value, valid_values)
            else:
                logger.warning(
                    f"Unrecognized parsed value: {parsed_value} for key {key} parsed by the LLM. "
                    f"It has type: {type(parsed_value)}."
                )
                close_matches = []

            if len(close_matches) == 0:
                if fallback_value:
                    logger.warning(
                        f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM. "
                        f"Will use default value: {fallback_value}."
                    )
                    parsed_json[key] = fallback_value
                else:
                    raise ValueError(
                        f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM."
                    )
            else:
                parsed_json[key] = close_matches[0]
    return parsed_json


def parse_and_check_json(
    raw_reply: str,
    expected_keys: Iterable[str],
    valid_values: Optional[Iterable[str]] = None,
    fallback_value: Optional[str] = None,
):
    if json_obj := parse_json(raw_reply):
        for key in expected_keys:
            if key not in json_obj:
                error = f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
                logging.error(error)
                raise OutputParserException(error)
        try:
            check_json_values(json_obj, valid_values, fallback_value)
        except ValueError as e:
            raise OutputParserException(e)
        return json_obj
    raise OutputParserException("JSON decoding error or JSON not found in output")


def get_outer_columns(all_columns, num_columns_each_end=10):
    if len(all_columns) <= num_columns_each_end * 2:
        return list(all_columns)
    return all_columns[:num_columns_each_end] + all_columns[-num_columns_each_end:]
