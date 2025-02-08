from pathlib import Path

from langchain_core.messages import HumanMessage

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings


def run_send_message(state: AutoMLAgentState, inference: AIInference):
    if state['solutions'][-1]['code'] is None:
        state['messages'] = [HumanMessage(
            content="Solution not found. Please try again.", name="AutoMLAgent")]
    else:
        message = inference.chat_completion(get_settings().prompts.automl.automl_chat.run_send_message.user.format(
            description=state['description'],
            metrics=state['metrics'],
            pipeline=state['pipeline'],
            code=state['solutions'][-1]['code']
        ))
        message = message.content + "\n\n" + "[[Code]]({code_url}) | [[Pipeline]]({pipeline_url}) | [[Submission]]({submission_url})".format(
            code_url="file://" +
                     str((Path(get_settings().config.result_dir) / "solution.py").resolve()),
            pipeline_url="file://" +
                         str((Path(get_settings().config.result_dir) /
                              "pipeline/pipeline.json").resolve()),
            submission_url="file://" +
                         str((Path(get_settings().config.result_dir) /
                              "submission.csv").resolve()),
        )
        state['messages'] = [HumanMessage(
            content=message, name="AutoMLAgent")]
    return state
