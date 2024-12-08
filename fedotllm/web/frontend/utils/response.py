from web.common.types import ResponseContent, BaseResponse
import streamlit as st
from typing_extensions import List, Dict
from web.frontend.components.st_graph import st_graph
from web.frontend.utils.utils import get_hash_key
from web.common.types import ResponseState

def create_expander_label(response: BaseResponse):
    if response.name:
        label = response.name.capitalize()
        if response.state == ResponseState.RUNNING:
            label = f":orange[:material/sprint:] {label}..."
        elif response.state == ResponseState.COMPLETE:
            label = f":green[:material/done_all:] {label}"
        # elif response.state == 'error':
        #     label = f":red[:material/close:] {label}"
        return label
    return ''

def render(response: ResponseContent):
    if isinstance(response, str):
        st.markdown(response)
    elif isinstance(response, List):
        for resp in response:
            render(resp)
    elif isinstance(response, BaseResponse):
        if response.name:
            with st.expander(label=create_expander_label(response), expanded=(response.state == ResponseState.RUNNING)):
                with st.empty():
                    render(response=response.content)
        elif response.content:
            render(response.content)
    elif isinstance(response, Dict):
        if response['type'] == 'graphviz':
            if isinstance(response['data'], str):
                st_graph(response['data'], key=get_hash_key('graphviz'))
    elif not response:
        return
    else:
        raise (ValueError("Unsupported response."))