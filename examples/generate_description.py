from fedot_llm.data.data import Dataset
from fedot_llm.language_models.llms import HuggingFaceLLM
from fedot_llm.language_models.actions import ModelAction
from pprint import pprint


def main():
    dataset = Dataset.load_from_path('../datasets/Health_Insurance')
    train = dataset.splits[0]
    model = HuggingFaceLLM(model_id="microsoft/Phi-3-mini-4k-instruct", max_new_tokens=500)
    action = ModelAction(model=model)
    column_descriptions = action.generate_all_column_description(split=train, dataset=dataset)
    train.set_column_descriptions(column_descriptions)
    pprint(train.get_column_descriptions())


if __name__ == '__main__':
    main()
