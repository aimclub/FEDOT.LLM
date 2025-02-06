from pathlib import Path

import streamlit as st
from fedotllm.settings.config_loader import get_settings
from fedotllm.web.backend.app import FedotAIBackend
from fedotllm.web.common.types import InitModel
from fedotllm.web.frontend.utils.utils import load_dataset
from streamlit_extras.grid import GridDeltaGenerator, grid
from fedotllm.web.frontend.localization import lclz


@st.dialog(lclz[st.session_state.lang]['PREVIEW'], width="large")
def split_preview(item):
    st.write(item.data)


def init_dataset():
    dataset_files_container = st.container(border=True)
    with dataset_files_container:
        st.header(lclz[st.session_state.lang]['INIT_DATASET_HEADER'])
        _render_file_uploader()
        _render_file_previews()


def _render_file_uploader():
    expander_state = not st.session_state.dataset
    with st.expander(lclz[st.session_state.lang]['CHOOSE_FILES'], expanded=expander_state):
        with st.form(key="dataset_files_form", border=False):
            st.file_uploader(
                "Choose dataset files",
                accept_multiple_files=True,
                key="file_uploader",
                label_visibility='collapsed'
            )
            st.form_submit_button(lclz[st.session_state.lang]['SUBMIT'], use_container_width=True,
                                  on_click=load_dataset)


def _render_file_previews():
    if not st.session_state.dataset:
        return

    with st.expander(lclz[st.session_state.lang]['FILES_PREVIEWS'], expanded=True):
        preview_grid = grid([1] * len(st.session_state.dataset.splits))
        for split in st.session_state.dataset.splits:
            preview_grid.button(
                split.name,
                on_click=split_preview,
                args=(split,),
                use_container_width=True
            )


def set_model(form_grid: GridDeltaGenerator):
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

    fedotai_backend: FedotAIBackend = st.session_state.fedotai_backend
    try:
        model = InitModel.model_construct(**params)
        fedotai_backend.init_model(model)
        st.toast(
            f"{lclz[st.session_state.lang]['SUCCESS_MODEL_SET']}:\n {st.session_state.model_name_input}",
            icon="✅")
        st.session_state.model = model
    except ValueError as e:
        form_grid.warning(str(e), icon="⚠")


def set_chat_model():
    if not st.session_state.model:
        form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')
        st.text_input(lclz[st.session_state.lang]['NAME'], placeholder="gpt-4o", value=get_settings().get("config.model") or "",
                      key="model_name_input",
                      disabled=bool(st.session_state.model))
        st.text_input(lclz[st.session_state.lang]['API_KEY'], placeholder=lclz[st.session_state.lang]['API_KEY'],
                      key="api_key",
                      value=get_settings().get("OPENAI_TOKEN") or "",
                      disabled=bool(st.session_state.model),
                      type='password')
        st.text_input(lclz[st.session_state.lang]['BASE_URL'], placeholder=lclz[st.session_state.lang]['BASE_URL'],
                      key="base_url",
                      value=get_settings().get("config.base_url") or "",
                      disabled=bool(st.session_state.model),
                      help=lclz[st.session_state.lang]['BASE_URL_HELP'])
        submit = st.button(label=lclz[st.session_state.lang]['SUBMIT'], use_container_width=True,
                           disabled=bool(st.session_state.model))
        if submit:
            set_model(form_grid)
            st.rerun()


def dwn_results():
    with st.container(border=True):
        st.header(lclz[st.session_state.lang]['RESULTS'])
        result_dir = Path(get_settings()['config']['result_dir'])
        solution_path = result_dir / "solution.py"
        submission_path = result_dir / "submission.csv"
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


def init_model():
    with st.container(border=True):
        st.header(lclz[st.session_state.lang]['INIT_LLM_MODEL_HEADER'])
        if not st.session_state.model:
            set_chat_model()
        else:
            st.write(
                f"{lclz[st.session_state.lang]['NAME']}: {st.session_state.model.name}")


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
