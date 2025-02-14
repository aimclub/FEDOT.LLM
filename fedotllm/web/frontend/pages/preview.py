import streamlit as st
from ..utils import get_user_uploaded_files
from pygwalker.api.streamlit import StreamlitRenderer
from ..localization import lclz

@st.fragment
def preview_dataset():
    """
    Displays a preview of the uploaded dataset in the Streamlit app.
    """
    st.header(lclz[st.session_state.lang]['FILES_PREVIEWS'])
    _, col2, _ = st.columns([1, 22, 1])
    with col2:
        file_options = get_user_uploaded_files()
        if st.session_state.output_file is not None:
            file_options.append("submission.csv")
        selected_file = st.selectbox(
            "Preview File",
            options=file_options,
            index=None,
            placeholder=lclz[st.session_state.lang]['SELECT_PREVIEW'],
            label_visibility="collapsed",
            key="_selected_file",
        )
        if not st.session_state.uploaded_files:
            st.info(lclz[st.session_state.lang]['PREVIEW_NOT_FOUND'], icon="ℹ️")
            return
        if selected_file is not None:
            st.markdown(
                f"""
            <div class="file-view-bar">
                <span class="file-view-label">Viewing File:</span> {selected_file}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.session_state.output_filename and selected_file == "submission.csv":
                output_file = st.session_state.output_file
                pyg_app = StreamlitRenderer(output_file)
            else:
                pyg_app =StreamlitRenderer(st.session_state.uploaded_files[selected_file]["df"])
            pyg_app.explorer()


def main():
    preview_dataset()


if __name__ == "__main__":
    main()
