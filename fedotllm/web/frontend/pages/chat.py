import asyncio

import streamlit as st

from fedotllm.web.backend.app import ask
from fedotllm.web.common.types import BaseResponse

from ..localization import lclz
from ..utils import generate_output_file, get_user_data_dir, render, save_all_files

def update_config_overrides():
    config_overrides = []
    if st.session_state.llm:
        config_overrides.append(f'llm.model={st.session_state.llm["name"]}')
        config_overrides.append(f'llm.base_url={st.session_state.llm["base_url"]}')
        config_overrides.append(f'llm.api_key={st.session_state.llm["api_key"]}')
    st.session_state.config_overrides = config_overrides


async def handle_predict(prompt):
    try:
        user_data_dir = get_user_data_dir()
        save_all_files(user_data_dir)
        update_config_overrides()
        gen_response = ask(
            prompt,
            task_path=user_data_dir,
            config_overrides=st.session_state.config_overrides,
            workspace=user_data_dir,
            lang=st.session_state.lang,
        )
        current_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": None})
        async for response in gen_response:
            if st.session_state.messages[current_idx]["content"]:
                if response.__eq__(st.session_state.messages[current_idx]["content"]):
                    continue
            else:
                st.session_state.messages[current_idx]["content"] = BaseResponse(
                    id=response.id
                )

            st.session_state.messages[current_idx]["content"] += response
            with st.container():
                render(response=st.session_state.messages[current_idx]["content"])
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise Exception(f"Error during prediction: {str(e)}")


def message_handler(message_container):
    prompt = st.session_state.task_description
    with message_container:
        try:
            add_user_message(prompt)
            assistant_placeholder = st.chat_message("assistant")
            with assistant_placeholder:
                answer_placeholder = st.empty()
                with answer_placeholder:
                    validate_model_and_dataset()
                    process_fedot_backend(prompt)
        except ValueError as e:
            add_assistant_error_message(str(e))
        except Exception as e:
            st.error(str(e), icon="⛔️")


def add_user_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)


def validate_model_and_dataset():
    if not st.session_state.llm and not st.session_state.uploaded_files:
        raise ValueError(lclz[st.session_state.lang]["NO_MODEL_AND_DATASET"])
    if not st.session_state.llm:
        raise ValueError(lclz[st.session_state.lang]["NO_MODEL"])
    if not st.session_state.uploaded_files:
        raise ValueError(lclz[st.session_state.lang]["NO_DATASET"])


def process_fedot_backend(prompt):
    with st.spinner(lclz[st.session_state.lang]["SPINNER_LABEL"]):
        try:
            asyncio.run(handle_predict(prompt))
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")


def add_assistant_error_message(content):
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.rerun()


def chat():
    if "task_running" not in st.session_state:
        st.session_state.task_running = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    message_container = st.container()
    if st.session_state.task_running:
        message_handler(message_container)
        generate_output_file()
        st.session_state.task_running = False
        st.rerun()
    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            render(message["content"])
