import os
import pytest
from pathlib import Path
import yaml
from fedotllm.utils.io import save_yaml, load_yaml, append_yaml

@pytest.fixture
def temp_yaml_file(tmp_path):
    """Fixture that provides a temporary YAML file path."""
    return tmp_path / "test.yaml"

@pytest.fixture
def sample_data():
    """Fixture that provides sample data for testing."""
    return {
        "string": "test",
        "number": 42,
        "boolean": True,
        "list": [1, 2, 3],
        "nested": {
            "key1": "value1",
            "key2": {
                "subkey": "subvalue"
            }
        }
    }

@pytest.fixture
def large_data():
    """Fixture that provides a larger dataset for testing."""
    return {
        "large_list": list(range(1000)),
        "nested_dicts": {
            f"key_{i}": {
                f"subkey_{j}": f"value_{i}_{j}"
                for j in range(10)
            }
            for i in range(10)
        }
    }

def test_save_yaml_basic(temp_yaml_file, sample_data):
    """Test basic YAML saving functionality."""
    save_yaml(sample_data, temp_yaml_file)
    assert temp_yaml_file.exists()
    with open(temp_yaml_file, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == sample_data

def test_save_yaml_large_file(temp_yaml_file, large_data):
    """Test saving large YAML data."""
    save_yaml(large_data, temp_yaml_file)
    assert temp_yaml_file.exists()
    loaded_data = load_yaml(temp_yaml_file)
    assert loaded_data == large_data

def test_load_yaml_nonexistent_file(temp_yaml_file):
    """Test loading from a non-existent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        load_yaml(temp_yaml_file)

def test_load_yaml_empty_file(temp_yaml_file):
    """Test loading from an empty YAML file."""
    temp_yaml_file.touch()
    assert load_yaml(temp_yaml_file) is None

def test_load_yaml_invalid_yaml(temp_yaml_file):
    """Test loading invalid YAML content."""
    temp_yaml_file.write_text("invalid: yaml: content: :")
    with pytest.raises(yaml.YAMLError):
        load_yaml(temp_yaml_file)

def test_append_yaml_to_empty_file(temp_yaml_file, sample_data):
    """Test appending to an empty file."""
    append_yaml(sample_data, temp_yaml_file)
    loaded_data = load_yaml(temp_yaml_file)
    assert loaded_data == sample_data

def test_append_yaml_to_existing_file(temp_yaml_file, sample_data):
    """Test appending to an existing YAML file by merging dictionaries."""
    initial_data = {"initial": "data", "unique": "value"}
    save_yaml(initial_data, temp_yaml_file)
    
    append_yaml(sample_data, temp_yaml_file)
    
    # Load the merged data
    loaded_data = load_yaml(temp_yaml_file)
    
    # Check that both dictionaries are merged
    expected_data = initial_data.copy()
    expected_data.update(sample_data)
    assert loaded_data == expected_data
    
    # Verify specific keys from both dictionaries exist
    assert loaded_data["initial"] == "data"
    assert loaded_data["unique"] == "value"
    assert loaded_data["string"] == "test"
    assert loaded_data["number"] == 42

def test_save_load_yaml_path_object(temp_yaml_file, sample_data):
    """Test that functions work correctly with Path objects."""
    path_obj = Path(temp_yaml_file)
    save_yaml(sample_data, path_obj)
    loaded_data = load_yaml(path_obj)
    assert loaded_data == sample_data

def test_save_yaml_creates_directory(tmp_path, sample_data):
    """Test that save_yaml creates intermediate directories if they don't exist."""
    nested_path = tmp_path / "nested" / "dirs" / "test.yaml"
    save_yaml(sample_data, nested_path)
    assert nested_path.exists()
    loaded_data = load_yaml(nested_path)
    assert loaded_data == sample_data

