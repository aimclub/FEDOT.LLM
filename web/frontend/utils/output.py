from __future__ import annotations
from fedot_llm.output import BaseFedotAIOutput
from langchain_core.runnables import Runnable
from typing import Dict, Any
from fedot_llm.ai.chains.legacy_chain import chains
import logging
import streamlit as st
from langchain_core.runnables.schema import StreamEvent
from streamlit.delta_generator import DeltaGenerator
    


class StreamlitFedotAIOutput(BaseFedotAIOutput):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(
            filename='llm.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
    
    def get_logging(self):
        log_msg = []
        def inner(event: StreamEvent):
            nonlocal log_msg
            if event['event'] == 'on_chat_model_end':
                if event.get('data', None) and 'chunk' not in event['data']:
                    if event['data'].get('input', None):
                        log_msg.append(
                            f"INPUT:\n{event['data']['input']['messages']}")
                    if event['data'].get('output', None):
                        log_msg.append(
                            f"ANSWER:\n{event['data']['output'].content}")
                    self.logger.debug(
                        f"{event['name']}\n" + '\n'.join(log_msg))
        return inner
    
    def get_display_progress(self, container: DeltaGenerator):
        progress = container.status("*Progress...*", expanded=True)
        progress_event = progress.empty()
        def inner(event: StreamEvent):
            nonlocal progress_event
            progress_container = progress_event.container()
            for step in steps:
                    if step.id in event['name']:
                        if event['event'] == 'on_chain_start':
                            step.status = 'Running'
                        elif event['event'] == 'on_chain_stream':
                            step.status = 'Streaming'
                        elif event['event'] == 'on_chain_end':
                            step.status = 'Сompleted'

            for step in steps:
                if step.status == 'Waiting':
                    progress_container.write(f"⌛️ {step.name}")
                if step.status == 'Running' or step.status == 'Streaming':
                    progress_container.write(f"🏃‍♂️‍➡️ {step.name}")
                elif step.status == 'Сompleted':
                    progress_container.write(f"✅ {step.name}")
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                progress.update(label="*Progress*", state="complete", expanded=False)
        return inner
    
    def get_display_analyze(self, container: DeltaGenerator):
        analyze = container.status("*Analyzing...*", expanded=True)
        analyze_event = analyze.empty()
        analyze_msg = ''
        def inner(event: StreamEvent):
            nonlocal analyze, analyze_msg, analyze_event
            if 'print' in event['tags']:
                analyze_msg += event['data'].get('chunk', '')
                analyze_event.write(analyze_msg)
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                analyze.update(label="*Analysis*", state="complete")
        return inner

    async def _chain_call(self, chain: Runnable, chain_input: Dict[str, Any]):
        placeholder = st.empty()
        answer_container = placeholder.container()
        save_log = self.get_logging()
        show_progress = self.get_display_progress(answer_container)
        analyze = self.get_display_analyze(answer_container)
        msg = 'Oh, you need to pick model first!'
        st.session_state.messages.append(
            {"role": "assistant", "content": msg})
        answer_container.write(msg)
        async for event in chain.astream_events(chain_input, version="v2"):
            save_log(event)
            show_progress(event)
            analyze(event)
            
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                for step in steps:
                    step.status = 'Waiting'
                st.session_state.messages.append(
                    {"role": "assistant", "content": placeholder})
                return event['data']['output']
