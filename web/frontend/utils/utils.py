from typing_extensions import Optional
from datetime import datetime
from hashlib import sha256
import streamlit as st
from web.frontend.utils.dataset_loader import StreamlitDatasetLoader

def get_hash_key(prefix: Optional[str]):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
    return sha256(f'{prefix}{date}'.encode()).hexdigest()

def load_dataset():
    files = st.session_state.file_uploader
    if files:
        st.session_state.dataset = StreamlitDatasetLoader.load(files=files)