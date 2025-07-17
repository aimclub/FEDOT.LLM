import functools
import os
import tempfile
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pytest import fixture

from fedotllm.configs.loader import load_config


class LLMTestConfig(BaseModel):
    model_name: str = "gpt-4.1"
    api_key: Optional[str] = None


class TestConfig(BaseModel):
    llm: LLMTestConfig
    fix_tries: int = 1
    predictor_init_kwargs: dict = Field(default_factory=dict)


@fixture
def config_file():
    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)

    config_content = {
        "llm": {
            "model_name": "gemini-2.0-flash",
            "api_key": "${oc.env:FEDOTLLM_LLM_API_KEY}",
        },
        "fix_tries": 5,
        "predictor_init_kwargs": {"timeout": 30},
    }

    env = {
        "FEDOTLLM_LLM_API_KEY": "sk-12345",
    }
    for key, val in env.items():
        os.environ[key] = val

    with open(config_path, "w") as f:
        yaml.safe_dump(config_content, f, indent=2)

    expected_config = config_content
    expected_config["llm"]["api_key"] = "sk-12345"

    yield config_path, deepcopy(expected_config)

    os.remove(config_path)


@fixture
def preset_files():
    presets = [{"fix_tries": 3}, {"predictor_init_kwargs": {"with_tune": False}}]
    preset_paths = []
    for preset in presets:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)  # Close the file descriptor to avoid PermissionError on Windows
        preset_paths.append(path)
        with open(path, "w") as f:
            yaml.safe_dump(preset, f, indent=2)

    yield preset_paths.copy(), deepcopy(presets)

    for path in preset_paths:
        os.remove(path)


def deep_merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deep_merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def test_load_config_with_env_variables(config_file):
    config_path, expected_config = config_file

    sut = load_config(schema=TestConfig, config_path=config_path)

    assert sut.model_dump() == expected_config


def test_override_config(config_file):
    config_path, expected_config = config_file
    overrides = ["llm.model_name='gpt-4o'", "fix_tries=1"]
    expected_config["llm"]["model_name"] = "gpt-4o"
    expected_config["fix_tries"] = 1

    sut = load_config(schema=TestConfig, config_path=config_path, overrides=overrides)

    assert sut.model_dump() == expected_config


def test_apply_presets(config_file, preset_files):
    config_path, expected_config = config_file
    preset_paths, presets = preset_files
    expected_config = functools.reduce(deep_merge, [expected_config, *presets])

    sut = load_config(schema=TestConfig, config_path=config_path, presets=preset_paths)

    assert sut.model_dump() == expected_config


def test_load_preset_from_standart_path(config_file):
    config_path, expected_config = config_file
    preset = {"fix_tries": 4}
    expected_config["fix_tries"] = 4
    preset_path = Path(files("fedotllm") / "configs" / "test.yaml")
    with preset_path.open("w") as f:
        yaml.safe_dump(preset, f, indent=2)

    sut = load_config(
        schema=TestConfig, config_path=config_path, presets="fedotllm:test"
    )

    assert sut.model_dump() == expected_config

    os.remove(preset_path)
