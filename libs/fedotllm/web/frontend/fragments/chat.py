import asyncio

import streamlit as st

from fedotllm.web.backend.app import FedotAIBackend
from fedotllm.web.common.types import BaseResponse
from fedotllm.web.frontend.components import st_write_str_stream
from fedotllm.web.frontend.utils import response as rp

from fedotllm.web.frontend.localization import lclz


async def handle_predict(prompt):
    fedotai_backend: FedotAIBackend = st.session_state.fedotai_backend
    st.session_state.messages.append(
        {"role": "assistant", "content": None})
    current_idx = len(st.session_state.messages) - 1
    gen_response = fedotai_backend.ask({'msg': prompt})
    async for response in gen_response:
        if st.session_state.messages[current_idx]["content"]:
            if response.__eq__(st.session_state.messages[current_idx]["content"]):
                continue
        else:
            st.session_state.messages[current_idx]["content"] = BaseResponse(
                id=response.id)

        st.session_state.messages[current_idx]["content"] += response
        with st.container():
            rp.render(
                response=st.session_state.messages[current_idx]["content"])


def message_handler(message_container):
    prompt = st.session_state.chat_input
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
    if not st.session_state.model and not st.session_state.dataset:
        raise ValueError(lclz[st.session_state.lang]['NO_MODEL_AND_DATASET'])
    if not st.session_state.model:
        raise ValueError(lclz[st.session_state.lang]['NO_MODEL'])
    if not st.session_state.dataset:
        raise ValueError(lclz[st.session_state.lang]['NO_DATASET'])


def process_fedot_backend(prompt):
    if st.session_state.fedotai_backend:
        with st.spinner(lclz[st.session_state.lang]['SPINNER_LABEL']):
            try:
                asyncio.run(handle_predict(prompt))
            except Exception as e:
                raise Exception(f"Error during prediction: {str(e)}")
    else:
        raise ValueError("FedotAI is None! Upload dataset files.")


def add_assistant_error_message(content):
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.rerun()


def chat():
    message_container = st.container()

    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            rp.render(message["content"])

    on_submit = st.chat_input(lclz[st.session_state.lang]['CHAT_INPUT_PLACEHOLDER'],
                              key='chat_input',
                              args=(message_container,),
                              disabled=st.session_state.chat_input_disable)
    if on_submit:
        message_handler(message_container)
        st.rerun()
