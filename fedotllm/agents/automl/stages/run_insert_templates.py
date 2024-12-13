from autoflake import fix_code
from fedotllm.agents.automl.templates.load_template import load_template, render_template

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.log import get_logger

logger = get_logger()


def run_insert_templates(state: AutoMLAgentState):
    logger.info("Running insert templates")
    code = state['codegen_sol']['code']
    fedot_config = state['fedot_config']
    predict_method = {
        'predict': "predict(features=input_data)",
        'forecast': "forecast(pre_history=input_data)",
        'predict_proba': "predict_proba(features=input_data)"
    }.get(fedot_config.predict_method)
    try:
        fedot_train = load_template("fedot_train.py")
        fedot_train = render_template(template=fedot_train,
                                      problem=f"{fedot_config.problem}",
                                      timeout=fedot_config.timeout,
                                      seed=fedot_config.seed,
                                      cv_folds=fedot_config.cv_folds,
                                      preset=f"'{fedot_config.preset.value}'",
                                      metric=f"'{fedot_config.metric.value}'")
        fedot_evaluate = load_template("fedot_evaluate.py")
        fedot_evaluate = render_template(template=fedot_evaluate,
                                         problem=f"{fedot_config.problem}",
                                         predict_method=predict_method)
        fedot_predict = load_template("fedot_predict.py")
        fedot_predict = render_template(template=fedot_predict,
                                        problem=f"{fedot_config.problem}",
                                        predict_method=predict_method)
        automl_temp = '\n'.join([fedot_train, fedot_evaluate, fedot_predict])
        code = code.replace(
            "from automl import train_model, evaluate_model", automl_temp)
    except Exception:
        logger.error("Model removed template anchors")
        state['solutions'][-1]['code'] = None
        return state

    code = fix_code(code, remove_all_unused_imports=True,
                    remove_unused_variables=True)
    logger.debug(f"Updated code: \n{code}")
    if "solutions" not in state:
        state['solutions'] = []
    state['solutions'].append({
        'code': code,
        'exec_result': state['codegen_sol']['exec_result'],
        'fix_tries': state['codegen_sol']['fix_tries']
    })
    return state
