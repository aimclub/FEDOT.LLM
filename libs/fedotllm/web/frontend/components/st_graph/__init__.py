from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

# Tell streamlit that there is a component called st_switcher,
# and that the code to display that component is in the "frontend" folder
frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "st_graph", path=str(frontend_dir)
)


# Create the python function that will be called
def st_graph(
        dot: Optional[str] = None,
        prev_dot: Optional[str] = None,
        key: Optional[str] = None
):
    """
    Add a descriptive docstring
    """
    component_value = _component_func(
        dot=dot,
        prev_dot=prev_dot,
        key=key
    )

    return component_value


def main():
    value = st_graph()
    st.write(value)


if __name__ == "__main__":
    main()
