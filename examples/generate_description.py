from fedotllm.data import Dataset
from fedotllm.web_api import WebAssistant
from fedotllm.actions import ModelAction
from pprint import pprint


def main():
    LLAMA8B = "http://10.32.2.2:8672/v1/chat/completions"
    dataset = Dataset.load_from_path('datasets/titanic')
    train = dataset.splits[0]
    model = WebAssistant(LLAMA8B, model_type='8b')
    action = ModelAction(model=model)
    column_descriptions = action.generate_all_column_description(split=train, dataset=dataset)
    train.set_column_descriptions(column_descriptions)
    pprint(train.get_column_descriptions())


if __name__ == '__main__':
    main()
