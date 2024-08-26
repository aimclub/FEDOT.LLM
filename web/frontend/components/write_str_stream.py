import streamlit as st
import time
from dataclasses import dataclass
@dataclass
class st_write_str_stream:
    string: str
    speed: float = 0.01
    def __post_init__(self):
        self._start()
        
    def _generator(self):
        for word in self.string:
            for char in word:
                yield char
                time.sleep(self.speed)
            time.sleep(self.speed*2)
    
    def _start(self):
        st.write_stream(self._generator())