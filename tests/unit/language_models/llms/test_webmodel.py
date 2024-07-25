from fedot_llm.language_models.llms import CustomWebLLM


class TestCustomWebModel:
    def test_generate_something(self):
        model = CustomWebLLM(
            url="http://10.32.2.2:8672/v1/chat/completions", model="8b"
        )
        print(model.generate(user_prompt="hi"))
