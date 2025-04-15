import fedotllm.prompting.prompts as prompts
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.llm import LiteLLMModel
from fedotllm.log import get_logger
from fedotllm.tabular import Dataset
from fedotllm.utils.parsers import extract_code

logger = get_logger()


def run_fix_solution(state: AutoMLAgentState, llm: LiteLLMModel, dataset: Dataset):
    logger.info("Running fix solution")
    codegen = state["codegen_sol"]
    assert codegen is not None, "Codegen solution is not found"
    solution = state["solutions"][-1]
    exec_result = solution["exec_result"]
    assert exec_result is not None, "Execution result is not found"
    files = "\n".join(
        [
            f"File: {file.name}\n"
            + "\n".join([f"- {col}" for col in file.data.columns])
            for file in dataset.splits
        ]
    )

    # Truncate stderr and stdout to 1000 characters
    stderr = exec_result.stderr
    if len(stderr) > 1000:
        stderr = stderr[-1000:]
    stdout = exec_result.stdout
    if len(stdout) > 1000:
        stdout = stdout[-1000:]

    fix_prompt = prompts.automl.fix_solution_prompt(
        user_instruction=state["description"],
        dataset_path=str(dataset.path.relative_to(dataset.path.cwd())),
        files=files,
        code_recent_solution=codegen["code"],
        stderr=stderr,
        stdout=stdout,
    )
    fixed_solution = llm.query(fix_prompt)
    extracted_code = extract_code(fixed_solution)
    if extracted_code:
        codegen["code"] = extracted_code
        codegen["fix_tries"] += 1
    else:
        raise ValueError("No code found in the response")
    return state
