from fedotllm.web.frontend import pages as pg
from fedotllm.web.frontend.pages.preview import main as preview
from fedotllm.web.frontend.pages.task import main as task
import streamlit as st
import os

st.set_page_config(
    page_title="FedotLLM",
    page_icon="fedotllm/web/frontend/static/images/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
css_file_path = os.path.join(current_dir, "style.css")

with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    pg.init_session()
    task()
    pg.chat()
    st.markdown("---", unsafe_allow_html=True)
    preview()


if __name__ == "__main__":
    main()
