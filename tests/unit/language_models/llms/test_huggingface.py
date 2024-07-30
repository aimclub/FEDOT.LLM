from fedot_llm.language_models.base import BaseLLM
from fedot_llm.language_models.llms import HuggingFaceLLM


class TestHuggingFace:
    def test_generate_something(self):
        model = HuggingFaceLLM(
            model_id="microsoft/Phi-3-mini-4k-instruct", max_new_tokens=500
        )
        print(model.generate(user_prompt="hi"))
