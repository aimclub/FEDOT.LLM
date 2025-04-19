# Initial Session state
DEFAULT_SESSION_VALUES = {
    "llm": {},
    "uploaded_files": {},
    "lang": "en",
    "messages": [
        {
            "role": "assistant",
            "content": "Hello! Pick a model, upload the dataset files and send me the task description.",
        }
    ],
    "prev_graph": None,
    "output_filename": None,
    "output_file": None,
    "task_description": None,
    "task_running": False,
}
