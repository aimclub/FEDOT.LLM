from fedot_llm.language_models.llms import CustomWebLLM


def main():
    model = CustomWebLLM(url='http://10.32.2.2:8672/v1/chat/completions', model='8b')
    print(model.generate(user_prompt='hi'))

if __name__ == "__main__":
    main()