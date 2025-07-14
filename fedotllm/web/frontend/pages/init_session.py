import os
from copy import deepcopy

import streamlit as st
from fedotllm.constants import DEFAULT_SESSION_VALUES
from fedotllm.configs.loader import load_config


def init_page():
    st.logo(
        image="fedotllm/web/frontend/static/images/fedot-llm-white.png",
        link="https://itmo-nss-team.github.io/",
    )
    st.title("FEDOT.LLM")


def initial_session_state():
    """
    Initial Session State
    """
    for key, default_value in DEFAULT_SESSION_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = (
                deepcopy(default_value)
                if isinstance(default_value, (dict, list))
                else default_value
            )
    if not st.session_state.llm:
        config = load_config()
        st.session_state.llm = {
            "name": config.llm.model_name,
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url,
        }


def _set_env(var: str):
    if not os.environ.get(var):
        print(f"No {var} in env")


_set_env("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FEDOT.LLM"


def init_session():
    init_page()
    initial_session_state()
