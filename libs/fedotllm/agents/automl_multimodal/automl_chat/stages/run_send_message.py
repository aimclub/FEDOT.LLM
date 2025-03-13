from pathlib import Path

from langchain_core.messages import HumanMessage

from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState
from fedotllm.llm.inference import AIInference
from fedotllm.settings.config_loader import get_settings

from fedotllm.log import get_logger
logger = get_logger()


def run_send_message(state: AutoMLMultimodalAgentState, inference: AIInference):

    message = inference.chat_completion(get_settings().prompts.automl.automl_chat.run_send_message.user.format(
        description=state['description'],
        metrics=state['metrics'],
        pipeline=state['pipeline'],
        code= " "#state['solutions'][-1]['code']
    ))
    
    #TODO: Fix markdown paths
    pipeline_img_path = str(Path(get_settings().config.output_dir) / "pipeline.png")

    pipeline_img = f"![alt text]({pipeline_img_path})"

    message = pipeline_img  + "\n\n" + message.content + "\n\n"
    message = message + "[[Code]]({code_url}) | [[Pipeline]]({pipeline_url}) | [[Submission]]({submission_url})".format(
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
        content=message, name="AutoMLMultimodalAgent")]
    return state