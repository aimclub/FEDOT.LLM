import streamlit as st
from streamlit_extras.grid import grid, GridDeltaGenerator

from fedotllm.settings.config_loader import get_settings
from fedotllm.web.backend.app import FedotAIBackend
from fedotllm.web.common.types import InitModel
from fedotllm.web.frontend.utils.utils import load_dataset


@st.dialog("Preview", width="large")
def split_preview(item):
    st.write(item.data)


def init_dataset():
    dataset_files_container = st.container(border=True)
    with dataset_files_container:
        st.header("Dataset Files")
        _render_file_uploader()
        _render_file_previews()


def _render_file_uploader():
    expander_state = not st.session_state.dataset
    with st.expander("Choose dataset files", expanded=expander_state):
        with st.form(key="dataset_files_form", border=False):
            st.file_uploader(
                "Choose dataset files",
                accept_multiple_files=True,
                key="file_uploader",
                label_visibility='collapsed'
            )
            st.form_submit_button("Submit", use_container_width=True,
                                  on_click=load_dataset)


def _render_file_previews():
    if not st.session_state.dataset:
        return

    with st.expander("Files previews", expanded=True):
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
        form_grid.warning('No name provided', icon="⚠")
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
            f"Successfully set up a model:\n {st.session_state.model_name_input}",
            icon="✅")
        st.session_state.model = model
    except ValueError as e:
        form_grid.warning(str(e), icon="⚠")


def set_chat_model():
    if not st.session_state.model:
        form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')
        st.text_input("model", placeholder="gpt-4o", value=get_settings().get("config.model") or "",
                      key="model_name_input",
                      disabled=bool(st.session_state.model))
        st.text_input("api_key", placeholder="Your api_key",
                      key="api_key",
                      value=get_settings().get("OPENAI_TOKEN") or "",
                      disabled=bool(st.session_state.model),
                      type='password')
        st.text_input("base url", placeholder="Optional base url",
                      key="base_url",
                      value=get_settings().get("config.base_url") or "",
                      disabled=bool(st.session_state.model),
                      help="Base URL for API requests. Only specify if using a proxy or service emulator.")
        submit = st.button(label="Submit", use_container_width=True,
                           disabled=bool(st.session_state.model))
        if submit:
            set_model(form_grid)
            st.rerun()


def init_model():
    with st.container(border=True):
        st.header("Model")
        if not st.session_state.model:
            set_chat_model()
        else:
            st.write(f"Name: {st.session_state.model.name}")


def side_bar():
    with st.sidebar:
        init_model()
        init_dataset()
