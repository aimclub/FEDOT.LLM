import streamlit as st

class st_spinner_element:
    def __init__(self, text = "In progress..."):
        self.text = text
        self._spinner = iter(self._start())
        next(self._spinner) 
        
    def _start(self):
        with st.spinner(self.text):
            yield 
    
    def end(self):
        next(self._spinner, None)