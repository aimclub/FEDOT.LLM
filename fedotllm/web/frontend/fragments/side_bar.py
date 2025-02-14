from pathlib import Path

import streamlit as st
from fedotllm.settings.config_loader import get_settings
from streamlit_extras.grid import GridDeltaGenerator, grid
from fedotllm.web.frontend.localization import lclz
from ..utils import file_uploader, get_user_data_dir, create_zip_file


@st.dialog('Preview', width="large")
def split_preview(selected_file):
    st.write(st.session_state.uploaded_files[selected_file]["df"])


def init_dataset():
    st.header(lclz[st.session_state.lang]['INIT_DATASET_HEADER'])
    if not st.session_state.uploaded_files:
        file_uploader()
        st.rerun()
    else:
        _render_file_previews()


def get_user_uploaded_files():
    files_name = []
    if st.session_state.uploaded_files is not None:
        uploaded_files = st.session_state.uploaded_files
        files_name = list(uploaded_files.keys())
    return files_name


def _render_file_previews():

    file_options = get_user_uploaded_files()

    selected_file = st.selectbox(
        "Preview File",
        options=file_options,
        index=None,
        placeholder="Select the file to preview",
        label_visibility="collapsed")

    if not st.session_state.uploaded_files:
        st.info("file not found yet.", icon="ℹ️")
        return
    if selected_file:
        split_preview(selected_file)
        selected_file = None


def set_llm(form_grid: GridDeltaGenerator):
    name = st.session_state.model_name_input
    if not name:
        form_grid.warning(lclz[st.session_state.lang]['NO_NAME'], icon="⚠")
        return
    api_key = st.session_state.api_key
    base_url = st.session_state.base_url
    params = {'name': name}
    if api_key:
        params['api_key'] = api_key
    if base_url:
        params['base_url'] = base_url
    st.toast(
        f"{lclz[st.session_state.lang]['SUCCESS_MODEL_SET']}:\n {st.session_state.model_name_input}",
        icon="✅")
    st.session_state.llm = params


def set_chat_model():
    if not st.session_state.llm:
        form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')
        st.text_input(lclz[st.session_state.lang]['NAME'], placeholder="gpt-4o", value=get_settings().get("config.model") or "",
                      key="model_name_input",
                      disabled=bool(st.session_state.llm))
        st.text_input(lclz[st.session_state.lang]['API_KEY'], placeholder=lclz[st.session_state.lang]['API_KEY'],
                      key="api_key",
                      value=get_settings().get("OPENAI_TOKEN") or "",
                      disabled=bool(st.session_state.llm),
                      type='password')
        st.text_input(lclz[st.session_state.lang]['BASE_URL'], placeholder=lclz[st.session_state.lang]['BASE_URL'],
                      key="base_url",
                      value=get_settings().get("config.base_url") or "",
                      disabled=bool(st.session_state.llm),
                      help=lclz[st.session_state.lang]['BASE_URL_HELP'])
        submit = st.button(label=lclz[st.session_state.lang]['SUBMIT'], use_container_width=True,
                           disabled=bool(st.session_state.llm))
        if submit:
            set_llm(form_grid)
            st.rerun()


def dwn_results():
    with st.container(border=True):
        st.header(lclz[st.session_state.lang]['RESULTS'])
        user_data_dir = get_user_data_dir()
        solution_path = user_data_dir / "solution.py"
        submission_path = user_data_dir / "submission.csv"
        pipeline_path = user_data_dir / "pipeline"

        if solution_path.exists():
            with open(solution_path, "rb") as file:
                btn = st.download_button(
                    label="Download code solution",
                    data=file,
                    file_name="solution.py",
                    mime="text/plain",
                )
        if submission_path.exists():
            with open(submission_path, "rb") as file:
                btn = st.download_button(
                    label="Download submission",
                    data=file,
                    file_name="submission.csv",
                    mime="text/csv",
                )
        if pipeline_path.exists():
            zip_file = create_zip_file(pipeline_path)
            with open(zip_file, "rb") as file:
                btn = st.download_button(
                    label="Download pipeline",
                    data=file,
                    file_name="pipeline.zip",
                    mime="application/zip",
                )


def init_model():
    with st.container(border=True):
        st.header(lclz[st.session_state.lang]['INIT_LLM_MODEL_HEADER'])
        if not st.session_state.llm:
            set_chat_model()
        else:
            st.write(
                f"{lclz[st.session_state.lang]['NAME']}: {st.session_state.llm['name']}")


def change_lang():
    if st.button(f'{st.session_state.lang.upper()}', type="primary", use_container_width=True):
        match (st.session_state.lang):
            case 'ru': st.session_state.lang = 'en'
            case 'en': st.session_state.lang = 'ru'
        st.session_state.messages[0] = {"role": "assistant",
                                        "content": lclz[st.session_state.lang]['GREETING_MSG']}
        st.rerun()


def side_bar():
    option_map = {
        0: ":material/language:",
        1: ":material/network_intelligence:",
        2: ":material/home_storage:",
        3: ":material/done_all:",
    }
    with st.sidebar:
        selection = st.segmented_control(
            lclz[st.session_state.lang]['MENU'],
            options=option_map.keys(),
            format_func=lambda option: option_map[option],
            selection_mode="single",
            default=0
        )
        match selection:
            case 0:
                change_lang()
            case 1:
                init_model()
            case 2:
                init_dataset()
            case 3:
                dwn_results()
            case _:
                change_lang()
