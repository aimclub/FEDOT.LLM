import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fedotllm.settings.config_loader import get_settings
from fedotllm.web.backend.app import FedotAIBackend
from fedotllm.web.frontend.localization import lclz

if 'lang' not in st.session_state.keys():
    st.session_state.lang = 'en'


def init_page():
    st.set_page_config(
        page_title="FedotLLM",
        page_icon="libs/fedotllm/web/frontend/static/images/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.logo(
        image='libs/fedotllm/web/frontend/static/images/fedot-llm-white.png',
        link='https://itmo-nss-team.github.io/'
    )
    st.title("💬 FEDOT.LLM")


def init_session_state():
    if "model" not in st.session_state.keys():
        st.session_state.model = None
    if "dataset" not in st.session_state.keys():
        st.session_state.dataset = None
    if 'lang' not in st.session_state.keys():
        st.session_state.lang = 'en'
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": lclz[st.session_state.lang]['GREETING_MSG']}
        ]
    if 'chat_input_disable' not in st.session_state:
        st.session_state.chat_input_disable = False
    if "fedotai_backend" not in st.session_state.keys():
        st.session_state.fedotai_backend = FedotAIBackend()
    if "model_name" not in st.session_state.keys():
        st.session_state.model_name = None
    if 'prev_graph' not in st.session_state.keys():
        st.session_state.prev_graph = None
    if "fresh_launch" not in st.session_state.keys():
        st.session_state.fresh_launch = True

    def _set_env(var: str):
        if not os.environ.get(var):
            print(f"No {var} in env")

    load_dotenv()
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "FEDOT.LLM"


def clear_results():
    if st.session_state.fresh_launch:
        result_dir = Path(get_settings()['config']['result_dir'])
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.fresh_launch = False


def init_session():
    init_page()
    init_session_state()
    clear_results()
