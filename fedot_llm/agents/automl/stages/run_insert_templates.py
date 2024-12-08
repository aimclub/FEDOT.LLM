from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.agents.automl.templates.load_template import insert_template, render_template
from autoflake import fix_code
from fedot_llm.log import get_logger

logger = get_logger()

def run_insert_templates(state: AutoMLAgentState):
    logger.info("Running insert templates")
    code = state['solutions'][-1]['code']
    fedot_config = state['fedot_config']
    try:
        code = insert_template(code, "main")
        code = render_template(template=insert_template(code, "fedot"), 
                            problem=f"'{fedot_config.problem.value}'",
                            timeout=fedot_config.timeout,
                            seed=fedot_config.seed,
                            cv_folds=fedot_config.cv_folds,
                            preset=f"'{fedot_config.preset.value}'",
                            metric=[metric.value for metric in fedot_config.metrics])
    except Exception:
        logger.error("Model removed template anchors")
        state['solutions'][-1]['code'] = None
        return state
    
    code = fix_code(code)
    logger.debug(f"Updated code: \n{code}")
    state['solutions'][-1]['code'] = code
    return state
