import streamlit as st

from ..localization import lclz
from ..utils import (
    create_zip_file,
    file_uploader,
    generate_output_file,
    get_user_data_dir,
    get_user_session_id,
)


def store_value(key):
    st.session_state[key] = st.session_state["_" + key]


def init_dataset():
    st.header(lclz[st.session_state.lang]["INIT_DATASET_HEADER"])
    file_uploader()


def set_llm(key):
    st.session_state.llm[key] = st.session_state["_llm_" + key]


def load_llm_value(key):
    st.session_state["_llm_" + key] = st.session_state.llm[key]


@st.fragment
def set_llm_name():
    load_llm_value("name")
    st.text_input(
        lclz[st.session_state.lang]["NAME"],
        placeholder="e.g., gpt-4o, claude-4, llama-3.1",
        key="_llm_name",
        on_change=set_llm,
        args=["name"],
    )


@st.fragment
def set_llm_api_key():
    load_llm_value("api_key")
    st.text_input(
        lclz[st.session_state.lang]["API_KEY"],
        placeholder="sk-...",
        key="_llm_api_key",
        on_change=set_llm,
        args=["api_key"],
        type="password",
    )


@st.fragment
def set_llm_base_url():
    load_llm_value("base_url")
    st.text_input(
        lclz[st.session_state.lang]["BASE_URL"],
        placeholder=lclz[st.session_state.lang]["BASE_URL"],
        key="_llm_base_url",
        help=lclz[st.session_state.lang]["BASE_URL_HELP"],
        on_change=set_llm,
        args=["base_url"],
    )


def change_lang():
    if st.button(
        f"{st.session_state.lang.upper()}", type="primary", use_container_width=True
    ):
        match st.session_state.lang:
            case "ru":
                st.session_state.lang = "en"
            case "en":
                st.session_state.lang = "ru"
        st.session_state.messages[0] = {
            "role": "assistant",
            "content": lclz[st.session_state.lang]["GREETING_MSG"],
        }
        st.rerun()


def dwn_results():
    with st.container(border=True):
        st.header(lclz[st.session_state.lang]["RESULTS"])
        user_data_dir = get_user_data_dir()
        solution_path = user_data_dir / "solution.py"
        submission_path = user_data_dir / "submission.csv"
        pipeline_path = user_data_dir / "pipeline"

        if solution_path.exists():
            with open(solution_path, "rb") as file:
                st.download_button(
                    label="Download code solution",
                    data=file,
                    file_name="solution.py",
                    mime="text/plain",
                )
        if submission_path.exists():
            generate_output_file()
            with open(submission_path, "rb") as file:
                st.download_button(
                    label="Download submission",
                    data=file,
                    file_name="submission.csv",
                    mime="text/csv",
                )
        if pipeline_path.exists():
            zip_file = create_zip_file(pipeline_path)
            with open(zip_file, "rb") as file:
                st.download_button(
                    label="Download pipeline",
                    data=file,
                    file_name="pipeline.zip",
                    mime="application/zip",
                )


def run_section():
    with st.sidebar:
        change_lang()
        st.header(lclz[st.session_state.lang]["INIT_LLM_MODEL_HEADER"])
        set_llm_name()
        set_llm_api_key()
        set_llm_base_url()
        init_dataset()


def main():
    get_user_session_id()
    run_section()


if __name__ == "__main__":
    main()
