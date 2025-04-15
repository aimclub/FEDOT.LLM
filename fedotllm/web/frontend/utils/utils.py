import shutil
import uuid
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing_extensions import Optional

BASE_DATA_DIR = Path("./user_data")


def get_user_data_dir():
    """
    Get or create a unique directory for the current user session.

    Returns:
        str: The path to the user's data directory.
    """
    if "user_data_dir" not in st.session_state:
        unique_dir = (
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}-{get_user_session_id()}"
        )
        st.session_state.user_data_dir = BASE_DATA_DIR / unique_dir
        st.session_state.user_data_dir.mkdir(parents=True, exist_ok=True)
    return st.session_state.user_data_dir


def get_user_session_id():
    """
    Get or generate a unique user session ID.

    Returns:
        str: A unique identifier for the current user session.
    """
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def clear_directory(directory: Path):
    """
    Clear the contents of a directory.

    Args:
        directory (Path): The directory to clear.
    """
    for item in directory.iterdir():
        try:
            if item.is_file():
                item.unlink()
            else:
                clear_directory(item)
                item.rmdir()
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")


def save_uploaded_file(file: UploadedFile, directory: Path):
    """
    Save an uploaded file to the specified directory.

    Args:
        file (UploadedFile): The file uploaded by the user.
        directory (str): The directory to save the file in.
    """
    file_path = directory / file.name
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def save_all_files(user_data_dir: Path):
    """
    When the task starts to run, save all the user's uploaded files to user's directory

    Args:
        user_data_dir (str): The directory path where user's files will be saved.
    """
    clear_directory(user_data_dir)
    for _, file_data in st.session_state.uploaded_files.items():
        save_uploaded_file(file_data["file"], user_data_dir)


def file_uploader():
    """
    Handle file uploads
    """
    uploaded_files = st.file_uploader(
        "Select the dataset",
        accept_multiple_files=True,
        label_visibility="collapsed",
        type=["csv", "xlsx"],
    )
    st.session_state.uploaded_files = {}
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        st.session_state.uploaded_files[file.name] = {"file": file, "df": df}


def generate_output_file():
    """
    Generate and store the output file after task completion.
    """
    st.session_state.output_filename = get_user_data_dir() / "submission.csv"
    if st.session_state.output_filename.exists():
        df = pd.read_csv(st.session_state.output_filename)
        st.session_state.output_file = df
    else:
        st.error(f"CSV file not found: {st.session_state.output_filename}")


def get_user_uploaded_files():
    files_name = []
    if st.session_state.uploaded_files is not None:
        uploaded_files = st.session_state.uploaded_files
        files_name = list(uploaded_files.keys())
    return files_name


def create_zip_file(model_path: Path):
    """
    Create a zip file of the model directory

    Args:
        model_path (str): Path to the model directory

    Returns:
        str: Path to the created zip file
    """
    if not model_path.exists():
        return None
    zip_path = f"{model_path}.zip"
    shutil.make_archive(model_path, "zip", model_path)
    return zip_path


def get_hash_key(prefix: Optional[str]):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
    return sha256(f"{prefix}{date}".encode()).hexdigest()
