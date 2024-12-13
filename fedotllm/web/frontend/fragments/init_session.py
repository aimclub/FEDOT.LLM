import os

import streamlit as st
from dotenv import load_dotenv

from fedotllm.web.backend.app import FedotAIBackend


def init_page():
    st.set_page_config(
        page_title="FedotLLM",
        page_icon="fedotllm/web/frontend/static/images/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.logo(
        image='fedotllm/web/frontend/static/images/fedot-llm-white.png',
        link='https://itmo-nss-team.github.io/'
    )
    st.title("💬 FEDOT.LLM")


def init_session_state():
    if "model" not in st.session_state.keys():
        st.session_state.model = None
    if "dataset" not in st.session_state.keys():
        st.session_state.dataset = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! Pick a model, upload the dataset files, send me the dataset description. "}
        ]
    if 'chat_input_disable' not in st.session_state:
        st.session_state.chat_input_disable = False
    if "fedotai_backend" not in st.session_state.keys():
        st.session_state.fedotai_backend = FedotAIBackend()
    if "model_name" not in st.session_state.keys():
        st.session_state.model_name = None
    if 'prev_graph' not in st.session_state.keys():
        st.session_state.prev_graph = None

    def _set_env(var: str):
        if not os.environ.get(var):
            print(f"No {var} in env")

    load_dotenv()
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "FEDOT.LLM"


def init_session():
    init_page()
    init_session_state()
