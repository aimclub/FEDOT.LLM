from fedot_llm.language_models.base import BaseLLM
from fedot_llm.language_models.llms import HuggingFaceLLM


class TestHuggingFace:
    def test_generate_something(self):
        model = HuggingFaceLLM(
            repo_id="microsoft/Phi-3-mini-4k-instruct"
        )
        print(model.generate(user_prompt="hi",  max_new_tokens=500))
