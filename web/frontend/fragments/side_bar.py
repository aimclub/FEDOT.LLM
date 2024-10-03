import streamlit as st
from langchain_ollama import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI
from streamlit_extras.grid import grid, GridDeltaGenerator

from fedot_llm.main import FedotAI
from web.backend.app import FedotAIBackend
from web.frontend.utils.utils import load_dataset


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

        _initialize_fedot_backend()


def _initialize_fedot_backend():
    st.session_state.fedot_backend = FedotAIBackend(
        fedot_ai=FedotAI(
            dataset=st.session_state.dataset,
            model=st.session_state.model
        )
    )


def set_model(form_grid: GridDeltaGenerator):
    name = st.session_state.model_name_input
    provider = st.session_state.model_provider_input.lower()

    if not name:
        form_grid.warning('No name provided', icon="⚠")
        return

    params = {'model': name, 'temperature': 0.0}

    if provider == 'openai':
        set_openai_model(form_grid, params)
    elif provider == 'ollama':
        set_ollama_model(form_grid, params)
    else:
        form_grid.error("Unknown provider", icon="⚠")


def set_openai_model(form_grid: GridDeltaGenerator, params: dict):
    openai_api_key = st.session_state.openai_api_key
    base_url = st.session_state.base_url

    if not openai_api_key:
        st.warning("Please enter your OpenAI API key!", icon="⚠")
        return

    params['api_key'] = openai_api_key
    if base_url:
        params['base_url'] = base_url

    try:
        st.session_state.model = ChatOpenAI(**params)
        st.session_state.model_name = st.session_state.model.model_name
        st.toast(
            f"Successfully set up a model:\n {st.session_state.model_name}",
            icon="✅")
    except ValueError as e:
        form_grid.warning(str(e), icon="⚠")


def set_ollama_model(form_grid: GridDeltaGenerator, params: dict):
    try:
        st.session_state.model = ChatOllama(**params)
        st.session_state.model_name = st.session_state.model.model
        st.toast(
            f"Successfully set up a model:\n {st.session_state.model_name}",
            icon="✅")
    except ValueError as e:
        form_grid.warning(str(e), icon="⚠")


def on_provider_selected(grid: GridDeltaGenerator):
    provider = st.session_state.model_provider_input
    match provider:
        case 'Ollama':
            grid.text_input("model", placeholder="llama3.1",
                            key="model_name_input",
                            disabled=bool(st.session_state.model))
        case 'OpenAI':
            grid.text_input("model", placeholder="gpt-3.5-turbo",
                            key="model_name_input",
                            disabled=bool(st.session_state.model))
            grid.text_input("token", placeholder="Your openai token",
                            key="openai_api_key",
                            disabled=bool(st.session_state.model),
                            type='password')
            grid.text_input("base url", placeholder="Optional base url",
                            key="base_url",
                            disabled=bool(st.session_state.model),
                            help="Base URL for API requests. Only specify if using a proxy or service emulator.")


def init_model():
    with st.container(border=True):
        st.header("Model")
        if not st.session_state.model:
            if not st.session_state.model:
                on_provider = st.selectbox("Select provider",
                                           options=['Ollama', 'OpenAI'],
                                           key='model_provider_input',
                                           placeholder='Select provider')
                form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')
                if on_provider:
                    on_provider_selected(form_grid)
                submit = st.button(label="Submit", use_container_width=True,
                                   disabled=bool(st.session_state.model))
                if submit:
                    set_model(form_grid)
        else:
            st.write(f"Name: {st.session_state.model_name}")


def side_bar():
    with st.sidebar:
        init_model()
        init_dataset()
