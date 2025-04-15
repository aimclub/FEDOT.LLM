import pytest
from fedotllm.utils.parsers import extract_code

@pytest.mark.unit
def test_extract_code_with_python_specifier():
    content = '''```python
def hello():
    print("Hello")
```'''
    expected = '''def hello():
    print("Hello")'''
    assert extract_code(content) == expected

@pytest.mark.unit
def test_extract_code_without_language_specifier():
    content = '''```
def hello():
    print("Hello")
```'''
    expected = '''def hello():
    print("Hello")'''
    assert extract_code(content) == expected
    
@pytest.mark.unit
def test_extract_code_without_starting_backticks():
    content = '''def hello():
    print("Hello")
```'''
    expected = '''def hello():
    print("Hello")'''
    assert extract_code(content) == expected

@pytest.mark.unit
def test_extract_code_with_indentation():
    content = '''    def hello():
        print("Hello")
    ```'''
    expected = '''def hello():
        print("Hello")'''
    assert extract_code(content) == expected

@pytest.mark.unit
def test_extract_code_with_empty_content():
    content = '''```python
```'''
    assert extract_code(content) is ''

@pytest.mark.unit
def test_extract_code_no_code_block():
    content = "Just some regular text without code block"
    assert extract_code(content) is None

@pytest.mark.unit
def test_extract_code_with_multiple_blocks():
    content = '''```python
def first():
    pass
```
Some text in between
```python
def second():
    pass
```'''
    expected = '''def first():
    pass'''
    assert extract_code(content) == expected  # Should return the first code block

@pytest.mark.unit
def test_extract_code_with_whitespace():
    content = '''```python    
    def hello():
        print("Hello")
    ```'''
    expected = '''def hello():
        print("Hello")'''
    assert extract_code(content) == expected 