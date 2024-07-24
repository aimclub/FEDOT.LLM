from fedot_llm.language_models.llms import HuggingFaceLLM

def main():
    model = HuggingFaceLLM(model_id="microsoft/Phi-3-mini-4k-instruct")
    print(model.generate(user_prompt='hi'))

if __name__ == "__main__":
    main()