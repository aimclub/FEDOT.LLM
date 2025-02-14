import streamlit as st
from ..utils import get_user_session_id, file_uploader
from ..localization import lclz


def store_value(key):
    st.session_state[key] = st.session_state["_" + key]

@st.fragment
def display_description():
    st.text_area(
        label=lclz[st.session_state.lang]['TASK_DESCRIPTION'],
        placeholder=lclz[st.session_state.lang]['TASK_DESCRIPTION_PLACEHOLDER'],
        key="_task_description",
        on_change=store_value,
        args=["task_description"],
        height=300,
    )


def init_dataset():
    st.header(lclz[st.session_state.lang]['INIT_DATASET_HEADER'])
    file_uploader()


def set_llm(key):
    st.session_state.llm[key] = st.session_state["_llm_" + key]
    

def load_llm_value(key):
    st.session_state["_llm_" + key] = st.session_state.llm[key]


@st.fragment
def set_llm_name():
    load_llm_value("name")
    st.text_input(lclz[st.session_state.lang]['NAME'], placeholder="gpt-4o",
                  key="_llm_name",
                  on_change=set_llm,
                  args=["name"])


@st.fragment
def set_llm_api_key():
    load_llm_value("api_key")
    st.text_input(lclz[st.session_state.lang]['API_KEY'], placeholder="gpt-4o",
                  key="_llm_api_key",
                  on_change=set_llm,
                  args=["api_key"],
                  type='password')


@st.fragment
def set_llm_base_url():
    load_llm_value("base_url")
    st.text_input(lclz[st.session_state.lang]['BASE_URL'], placeholder=lclz[st.session_state.lang]['BASE_URL'],
                  key="_llm_base_url",
                  help=lclz[st.session_state.lang]['BASE_URL_HELP'],
                  on_change=set_llm,
                  args=["base_url"])


def change_lang():
    if st.button(f'{st.session_state.lang.upper()}', type="primary", use_container_width=True):
        match (st.session_state.lang):
            case 'ru':
                st.session_state.lang = 'en'
            case 'en':
                st.session_state.lang = 'ru'
        st.session_state.messages[0] = {"role": "assistant",
                                        "content": lclz[st.session_state.lang]['GREETING_MSG']}
        st.rerun()

def run_section():
    st.write(st.session_state.llm)
    st.header(lclz[st.session_state.lang]['RUN_FEDOTLLM'])
    _, mid, _ = st.columns([1, 22, 1], gap="large")
    with mid:
        change_lang()
        col2, col3, col4 = st.columns([11.9, 0.2, 11.9], gap="large")
        with col2:
            st.header(lclz[st.session_state.lang]['INIT_LLM_MODEL_HEADER'])
            set_llm_name()
            set_llm_api_key()
            set_llm_base_url()
            display_description()
        with col3:
            st.html(
                """
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 2px solid rgba(49, 51, 63, 0.2);
                            height: 700px;
                            margin: auto;
                        }
                    </style>
                """
            )
        with col4:
            init_dataset()
        


def main():
    get_user_session_id()
    run_section()


if __name__ == "__main__":
    main()
