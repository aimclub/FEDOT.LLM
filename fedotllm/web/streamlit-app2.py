from fedotllm.web.frontend import pages as pg
from fedotllm.web.frontend.pages.side_bar import main as side_bar
from fedotllm.web.frontend.localization import lclz
import streamlit as st
import os

st.set_page_config(
    page_title="FedotLLM",
    page_icon="fedotllm/web/frontend/static/images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
css_file_path = os.path.join(current_dir, "style.css")

try:
    with open(css_file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning(f"Custom stylesheet not found at {css_file_path}")


def store_query(key):
    st.session_state[key] = st.session_state["_" + key]


def query_input():
    if st.chat_input(
        placeholder=lclz[st.session_state.lang]["TASK_DESCRIPTION_PLACEHOLDER"],
        key="_task_description",
        on_submit=store_query,
        args=["task_description"],
    ):
        st.session_state.task_running = True
        st.rerun()


def main():
    pg.init_session()
    side_bar()
    pg.chat()
    query_input()


if __name__ == "__main__":
    main()
