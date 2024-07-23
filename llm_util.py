import os
import json

def run_model_call(model, system, context, task):
    """Run a prompt on model
    """
    model.set_sys_prompt(system)
    model.add_context(context)
    response = model(task, as_json=True)
    return response

def run_model_multicall(model, prompts):
    """Run a list of prompts on web model
    """
    responses = {}
    for task in prompts:
        response = run_model_call(
            model = model,
            system = prompts[task]["system"],
            context = prompts[task]["context"],
            task = prompts[task]["task"]
        )
        responses[task] = response
    return responses

def process_model_responses(responses, operations):
    for key in operations:
        responses[key] = operations[key](responses[key])
    return responses

def process_model_responses_for_v1(responses):
    operations = {
        "categorical_columns": lambda x : x.split("\n"),
        "task_type": lambda x : x.lower()
    }
    responses = process_model_responses(responses, operations)
    return responses

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