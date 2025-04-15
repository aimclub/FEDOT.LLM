from pathlib import Path
from typing import Union

import yaml


def read_file_safely(filename: Path) -> Union[str, None]:
    try:
        return filename.read_text()
    except UnicodeDecodeError:
        return None


def file_preview(filename: Path, max_chars: int = 100) -> str:
    content = read_file_safely(filename)
    if content is not None:
        truncated_contents = content[:max_chars].strip()
        if len(content) > max_chars:
            truncated_contents += "..."
        return f"File:\n\n{filename} Truncated Content:\n{truncated_contents}\n\n"
    return ""


def save_yaml(data: dict, filename: Path):
    # Create parent directories if they don't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f)


def append_yaml(data: dict, filename: Path):
    # Create parent directories if they don't exist
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if file exists, otherwise use empty dict
    existing_data = {}
    if filename.exists() and filename.stat().st_size > 0:
        existing_data = load_yaml(filename)

    # Merge the dictionaries
    if existing_data is None:
        existing_data = {}
    existing_data.update(data)

    # Save merged data
    with open(filename, "w") as f:
        yaml.dump(existing_data, f)


def load_yaml(filename: Path) -> dict:
    with open(filename, "r") as f:
        return yaml.safe_load(f)
