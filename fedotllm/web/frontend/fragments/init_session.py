import os

import streamlit as st
from dotenv import load_dotenv
from fedotllm.constants import DEFAULT_SESSION_VALUES
from copy import deepcopy


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


def initial_session_state():
    """
    Initial Session State
    """
    for key, default_value in DEFAULT_SESSION_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = (
                deepcopy(default_value) if isinstance(
                    default_value, (dict, list)) else default_value
            )


def _set_env(var: str):
    if not os.environ.get(var):
        print(f"No {var} in env")


load_dotenv()
_set_env("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FEDOT.LLM"


def init_session():
    init_page()
    initial_session_state()
