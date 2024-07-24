import os
import json

class ModelAction():
    
    def __init__(self, model) -> None:
        self.model = model

    def run_model_call(self, system, context, task):
        """Run a prompt on model
        """
        self.model.set_sys_prompt(system)
        self.model.set_context(context)
        response = self.model(task, as_json=True)
        return response

    def run_model_multicall(self, prompts):
        """Run a list of prompts on web model
        """
        responses = {}
        for task in prompts:
            response = self.run_model_call(
                system = prompts[task]["system"],
                context = prompts[task]["context"],
                task = prompts[task]["task"]
            )
            responses[task] = response
        return responses
    
    @staticmethod
    def process_model_responses(responses, operations):
        for key in operations:
            responses[key] = operations[key](responses[key])
        return responses

    @classmethod
    def process_model_responses_for_v1(cls, responses):
        operations = {
            "categorical_columns": lambda x : x.split("\n"),
            "task_type": lambda x : x.lower()
        }
        responses = cls.process_model_responses(responses, operations)
        return responses

    @staticmethod
    def save_model_responses(responses, path):
        with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
            json.dump(responses, json_file)



# def run_model_multicall(model, tokenizer, generation_config, prompts):
#     """Run all prompts on local model

#     TODO: transform to an interaction with local model helper class
#     """
    
#     responses = {}
#     for task in prompts:
#         messages = [
#             {"role": "system", "content": prompts[task]["system"]},
#             {"role": "context", "content": prompts[task]["context"]},
#             {"role": "user", "content": prompts[task]["task"]},
#         ]
        
#         input_ids = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(model.device)
        
#         outputs = model.generate(
#             input_ids,
#             **generation_config
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         responses[task] = tokenizer.decode(response, skip_special_tokens=True)

#     return responses