class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error."""


class ContextWindowExceededError(Exception):
    """Exception raised when the context window is exceeded."""
