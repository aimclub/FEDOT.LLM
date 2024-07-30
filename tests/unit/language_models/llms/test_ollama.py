from pprint import pprint

from fedot_llm.data import Dataset
from fedot_llm.language_models.actions import ModelAction
from fedot_llm.language_models.llms import OllamaLLM


class TestOllamaLLM:
    def test_generate_something(self):
        model = OllamaLLM(model="llama3")
        print(model.generate(user_prompt="hi"))

    def test_generate_description(self):
        dataset = Dataset.load_from_path("datasets/Health_Insurance")
        train = dataset.splits[0]
        model = OllamaLLM(model="llama3")
        action = ModelAction(model=model)
        column_descriptions = action.generate_all_column_description(
            split=train, dataset=dataset
        )
        train.set_column_descriptions(column_descriptions)
        pprint(column_descriptions)
