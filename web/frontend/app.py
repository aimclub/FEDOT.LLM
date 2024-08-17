import streamlit as st
from fedot_llm.main import FedotAI
from fedot_llm.chat import init_chat_model
from web.frontend.utils import StreamlitFedotAIOutput, StreamlitDatasetLoader
from streamlit_extras.grid import grid, GridDeltaGenerator
from web.frontend.components import st_write_str_stream
from web.backend.app import FedotAIBackend
from web.common.types import BaseResponse
from typing_extensions import Union, List
import asyncio

st.set_page_config(
    page_title="FedotLLM",
    page_icon="web/frontend/static/images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_dataset():
    files = st.session_state.file_uploader
    if files:
        st.session_state.dataset = StreamlitDatasetLoader.load(files=files)


@st.dialog("Preview", width="large")
def split_preview(item):
    st.write(item.data)


def crate_expander_label(response: BaseResponse) -> str:
    if response.name:
        label = response.name.capitalize()
        if response.state == 'running':
            label = f":orange[:material/sprint:] {label}..."
        elif response.state == 'complete':
            label = f":green[:material/done_all:] {label}"
        elif response.state == 'error':
            label = f":red[:material/close:] {label}"
        return label
    return ''


def process_response(response: Union[None, str, List[BaseResponse], BaseResponse]):
    if isinstance(response, str):
        st.markdown(response)
    elif isinstance(response, List):
        for resp in response:
            process_response(resp)
    elif isinstance(response, BaseResponse):
        if response.name:
            with st.expander(label=crate_expander_label(response), expanded=(response.state == 'running')):
                st.empty()
                process_response(response=response.content)
        elif response.content:
            process_response(response.content)
    elif not response:
        return
    else:
        raise (ValueError("Unsupported response."))


async def handle_predict(prompt):
    fedot_backend: FedotAIBackend = st.session_state.fedot_backend
    st.session_state.messages.append(
        {"role": "assistant", "content": BaseResponse()})
    current_idx = len(st.session_state.messages) - 1
    gen_response = fedot_backend.get_predict({'msg': prompt})
    async for response in gen_response:
        st.session_state.messages[current_idx]["content"] += response
        with st.container():
            process_response(response=st.session_state.messages[current_idx]["content"])


def message_handler(message_container):
    prompt = st.session_state.chat_input
    with message_container:
        try:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            assistant_placeholder = st.chat_message(
                "assistant")
            with assistant_placeholder:
                answer_placeholder = st.empty()
                with answer_placeholder:
                    if not st.session_state.model:
                        msg = 'Oh, you need to pick model first!'
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg})
                        st_write_str_stream(msg)
                        raise ValueError("Fill set model form")
                    if not st.session_state.dataset:
                        msg = 'Oh, you need to upload the dataset files first!'
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg})
                        st_write_str_stream(msg)
                        raise ValueError("Upload dataset files")
                    if st.session_state.fedot_backend:

                        with st.spinner("Give me a moment..."):
                            try:
                                # response_container = st.container()
                                # with response_container:
                                asyncio.run(handle_predict(prompt))
                            except Exception as e:
                                st.error(e, icon="⛔️")
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": 'Oh, something went wrong!'})
                        st_write_str_stream(
                            'FedotAI is None!')
                        raise ValueError("Upload dataset files")
        except Exception as e:
            st.error(e, icon="⛔️")


def chat_fragment():
    message_container = st.container()

    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            process_response(message["content"])

    on_submit = st.chat_input("Enter a prompt here...",
                  key='chat_input',
                  args=(message_container,),
                  disabled=st.session_state.chat_input_disable)
    if on_submit:
        message_handler(message_container)


def init_dataset():
    dataset_files_container = st.container(border=True)
    with dataset_files_container:
        st.header("Dataset Files")
        with st.expander("Choose dataset files", expanded=(not (st.session_state.dataset))):
            with st.form(key="dataset_files_form", border=False):
                files = st.file_uploader("Choose dataset files",
                                         accept_multiple_files=True,
                                         key="file_uploader",
                                         label_visibility='collapsed')
                st.form_submit_button("Submit", use_container_width=True,
                                      on_click=load_dataset)

        if st.session_state.dataset:
            with st.expander("Files previews", expanded=bool(st.session_state.dataset)):
                # columns = st.columns(len(st.session_state.dataset.splits))
                preview_grid = grid([1 * len(st.session_state.dataset.splits)])
                for split in st.session_state.dataset.splits:
                    preview_grid.button(split.name, on_click=split_preview,
                                        args=(split,), use_container_width=True)
                st.session_state.fedot_backend = FedotAIBackend(fedotAI=FedotAI(dataset=st.session_state.dataset,
                                                                                model=st.session_state.model,
                                                                                output=StreamlitFedotAIOutput()
                                                                                ))


def init_session_state():
    if "model" not in st.session_state.keys():
        st.session_state.model = None
    if "dataset" not in st.session_state.keys():
        st.session_state.dataset = None
    # if "fedot_ai" not in st.session_state.keys():
    #     st.session_state.fedot_ai = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! Pick a model, upload the dataset files, send me the dataset description. "}
        ]
    if 'chat_input_disable' not in st.session_state:
        st.session_state.chat_input_disable = False
    if "fedotai_backend" not in st.session_state.keys():
        st.session_state.fedotai_backend = None


def set_model(form_grid: GridDeltaGenerator):
    name = st.session_state.model_name
    provider = st.session_state.model_provider
    if name and provider:
        try:
            st.session_state.model = init_chat_model(
                model=name,
                model_provider=provider,
                temperature=0)
            st.toast(
                f"Successfully set up a model:\n {st.session_state.model.model}", icon="✅")

        except ValueError as e:
            form_grid.warning(e)
    elif not name and not provider:
        form_grid.warning('All fields are required', icon="⚠")
    elif not name:
        form_grid.warning('No name provided', icon="⚠")
    elif not provider:
        form_grid.warning('No provider specified', icon="⚠")
    else:
        form_grid.error("Unknown error", icon="⚠")


def init_model():
    with st.form("set_model_form"):
        form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')
        form_grid.header(
            "Set Model",
            help="For more information [visit](https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.base.init_chat_model.html)")
        form_grid.text_input("model", placeholder="llama3.1",
                             key="model_name", disabled=bool(st.session_state.model))
        form_grid.text_input("model provider", placeholder="ollama",
                             key="model_provider", disabled=bool(st.session_state.model))
        form_grid.form_submit_button(label="Submit", use_container_width=True, on_click=set_model, args=[
            form_grid], disabled=bool(st.session_state.model))


def side_bar():
    with st.sidebar:
        init_model()
        init_dataset()


def main():
    st.logo(
        image='web/frontend/static/images/fedot-llm-white.png',
        link='https://itmo-nss-team.github.io/'
    )
    st.title("💬 FEDOT.LLM")
    init_session_state()
    side_bar()
    chat_fragment()


if __name__ == "__main__":
    main()
