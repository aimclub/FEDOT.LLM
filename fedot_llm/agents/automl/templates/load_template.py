import re
from pathlib import Path
from typing import Union, Set, Optional

Path.templates_root = Path(__file__).parent

def render_sub_templates(text: str, loaded_templates: Optional[Set[Path]] = None) -> str:
    """
    Processes the input text by replacing template placeholders with their corresponding content.
    It also aggregates import statements from all sub-templates and inserts them appropriately.

    Args:
        text (str): The input text containing template placeholders.
        loaded_templates (Optional[Set[Path]]): Set of already loaded templates to prevent circular references.

    Returns:
        str: The processed text with all templates rendered and imports aggregated.
    """
    if loaded_templates is None:
        loaded_templates = set()

    sub_templates = re.findall(r"<%%\s*(.*?)\s*%%>", text)
    sub_imports = set()

    for sub_template in sub_templates:
        sub_template_path = sub_template.strip().lower()
        sub_template_content = load_template(sub_template_path, loaded_templates)

        # Extract and remove import statements
        imports = _extract_imports(sub_template_content)
        sub_imports.update(imports)
        sub_template_content = _remove_imports(sub_template_content, imports).strip()

        # Replace the placeholder with the template content, preserving indentation
        text = _replace_placeholder_with_content(text, sub_template, sub_template_content)

    # Aggregate and insert imports
    if sub_imports:
        imports_str = "\n".join(sorted(sub_imports))
        text = _insert_imports(text, imports_str)

    return text.strip()

def _extract_imports(content: str) -> Set[str]:
    """
    Extracts import statements from the given content.

    Args:
        content (str): The content to extract imports from.

    Returns:
        Set[str]: A set of import statements.
    """
    import_pattern = re.compile(
        r"^(import\s+\S+|from\s+\S+\s+import\s+\S+)",
        re.MULTILINE
    )
    imports = set(match.group(1) for match in import_pattern.finditer(content))
    return imports

def _remove_imports(content: str, imports: Set[str]) -> str:
    """
    Removes the specified import statements from the content.

    Args:
        content (str): The original content.
        imports (Set[str]): The import statements to remove.

    Returns:
        str: The content without the specified imports.
    """
    for imp in imports:
        content = content.replace(imp, "")
    return content

def _replace_placeholder_with_content(text: str, placeholder: str, content: str) -> str:
    """
    Replaces the template placeholder in the text with the actual content,
    preserving the original indentation.

    Args:
        text (str): The original text containing the placeholder.
        placeholder (str): The name of the placeholder to replace.
        content (str): The content to insert in place of the placeholder.

    Returns:
        str: The text with the placeholder replaced by the content.
    """
    pattern = re.compile(rf"(?P<indent>^\s*)<%%\s*{re.escape(placeholder)}\s*%%>", re.MULTILINE)
    match = pattern.search(text)
    if match:
        indent = match.group("indent")
        indented_content = "\n".join(
            f"{indent}{line}" if line.strip() else line
            for line in content.splitlines()
        )
        text = pattern.sub(indented_content, text, count=1)
    else:
        # Fallback if pattern does not match
        text = text.replace(f"<%%{placeholder}%%>", content)
    return text

def _insert_imports(text: str, imports_str: str) -> str:
    """
    Inserts aggregated import statements into the text after initial comments.

    Args:
        text (str): The original text.
        imports_str (str): The aggregated import statements as a single string.

    Returns:
        str: The text with imports inserted.
    """
    comment_pattern = re.compile(r'^(\s*#.*\n)*')
    match = comment_pattern.match(text)
    if match:
        comments = match.group(0)
        rest_of_text = text[len(comments):]
        text = f"{comments}{imports_str}\n{rest_of_text}"
    else:
        text = f"{imports_str}\n{text}"
    return text

def load_template(template_path: Union[str, Path], loaded_templates: Optional[Set[Path]] = None) -> str:
    """
    Loads and processes a template file by resolving its sub-templates,
    aggregating import statements, and preserving indentation.

    Args:
        template_path (Union[str, Path]): Path to the template file.
        loaded_templates (Optional[Set[Path]]): Set of already loaded templates to prevent circular dependencies.

    Returns:
        str: The processed template content with aggregated imports and properly indented inserted templates.

    Raises:
        ValueError: If a circular template reference is detected.
        FileNotFoundError: If the template file does not exist.
    """
    if loaded_templates is None:
        loaded_templates = set()

    template_path = Path(template_path)
    if not template_path.is_absolute():
        template_path = Path.templates_root / template_path

    if template_path in loaded_templates:
        raise ValueError(f"Circular template reference detected: {template_path}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    loaded_templates.add(template_path)
    try:
        file_content = template_path.read_text()
        processed_content = render_sub_templates(file_content, loaded_templates)
    finally:
        loaded_templates.remove(template_path)

    return processed_content.strip()

def render_template(template: str, **kwargs) -> str:
    """
    Replaces placeholders in the template with corresponding keyword arguments.

    Args:
        template (str): The template string containing placeholders like {% var %}.
        **kwargs: Variable names and their values to replace in the template.

    Returns:
        str: The rendered template with all placeholders replaced.

    Raises:
        ValueError: If any placeholder does not have a corresponding keyword argument.
    """
    import re

    pattern = re.compile(r"{%\s*(\w+)\s*%}")
    not_found = []

    def replace_match(match):
        var_name = match.group(1)
        if var_name in kwargs:
            value = kwargs[var_name]
            return str(value)
        else:
            not_found.append(var_name)
            return match.group(0)  # Keep the placeholder unchanged

    rendered = pattern.sub(replace_match, template)

    if not_found:
        raise ValueError(f"Variables not found for placeholders: {not_found}")

    return rendered

def insert_template(code: str, template: str) -> str:
    """
    Inserts a sub-template into the main code by replacing the placeholder.

    The placeholder format in the code should be:
    ### template ###
    ...sub-template content...
    ### template ###

    After insertion, the entire code is processed through `render_templates` to handle
    any nested templates.

    Args:
        code (str): The main code where the template will be inserted.
        template (str): The name of the template to insert.

    Returns:
        str: The code with the template inserted and all templates rendered.

    Raises:
        ValueError: If the template placeholder is not found in the code.
    """
    # Normalize the template name
    template = template.lower().strip()
    sub_imports: Set[str] = set()
    
    # Define regex pattern to match the placeholder with optional whitespace and multiline content
    # Capturing groups:
    # 1. Leading whitespace before the first ###
    # 2. The entire placeholder including the content to replace
    pattern = re.compile(
        rf"(?P<indent>^[ \t]*)###\s*{re.escape(template)}\s*###.*?###\s*{re.escape(template)}\s*###",
        re.IGNORECASE | re.DOTALL | re.MULTILINE
    )

    def replacement(match: re.Match) -> str:
        """
        Replacement function to load and insert the sub-template content with preserved indentation.

        Args:
            match (re.Match): The regex match object.

        Returns:
            str: The loaded and correctly indented sub-template content.

        Raises:
            ValueError: If loading the template fails.
        """
        try:
            leading_whitespace = match.group("indent") or ""
            
            # Load the template content using `load_template`, which internally calls `render_templates`
            loaded_content = load_template(template, sub_imports)
            
            # Extract and remove import statements from the loaded content
            extracted_imports = _extract_imports(loaded_content)
            sub_imports.update(extracted_imports)
            loaded_content = _remove_imports(loaded_content, extracted_imports).strip()
            
            # Indent each line of the loaded content to match the placeholder's indentation
            indented_content = "\n".join(
                f"{leading_whitespace}{line}" if line.strip() else line
                for line in loaded_content.splitlines()
            )
            
            return indented_content
        except Exception as e:
            raise ValueError(f"Failed to load and render template '{template}': {e}") from e

    # Perform the substitution using the replacement function
    new_code, count = pattern.subn(replacement, code)

    if count == 0:
        raise ValueError(f"Template placeholder '### {template} ###' not found in the code.")

    # After insertion, ensure that any nested templates within the code are rendered
    rendered_code = render_sub_templates(new_code, sub_imports)
    
    if sub_imports:
        imports_str = "\n".join(sorted(sub_imports))
        rendered_code = _insert_imports(rendered_code, imports_str)

    return rendered_code.strip()