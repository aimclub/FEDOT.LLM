from datetime import datetime
from hashlib import sha256
from pathlib import Path

import streamlit as st
from fedotllm.web.backend.app import FedotAIBackend
from fedotllm.web.frontend.utils.dataset_loader import StreamlitDatasetLoader
from typing_extensions import Optional

BASE_DATA_DIR = Path("./user_data")


def get_user_data_dir():
    """
    Get or create a unique directory for the current user session.

    Returns:
        str: The path to the user's data directory.
    """
    if "user_data_dir" not in st.session_state:
        unique_dir = st.session_state.user_session_id
        st.session_state.user_data_dir = BASE_DATA_DIR / unique_dir
        st.session_state.user_data_dir.mkdir(exist_ok=True)
    return st.session_state.user_data_dir


def get_hash_key(prefix: Optional[str]):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
    return sha256(f'{prefix}{date}'.encode()).hexdigest()


def load_dataset():
    files = st.session_state.file_uploader
    if files:
        st.session_state.dataset = StreamlitDatasetLoader.load(files=files)
        fedotai_backend: FedotAIBackend = st.session_state.fedotai_backend
        fedotai_backend.init_dataset(st.session_state.dataset)
