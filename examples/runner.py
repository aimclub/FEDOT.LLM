import sys
import os
import shutil
import subprocess
import re
import logging
from pathlib import Path
from fedotllm.main import FedotAI
from fedotllm.output import JupyterOutput
from fedotllm.llm import AIInference
from examples.kaggle import download_from_kaggle, submit_to_kaggle
from golem.core.dag.graph_utils import graph_structure
from fedot import Fedot

def setup_logging(log_file: str = None):
    """Setup logging configuration."""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file)],
            force=True
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Disable logging if no file

def log_print(message: str):
    """Print message and optionally log to file."""
    print(message)
    if logging.getLogger().handlers:
        logging.info(message)

def save_report(output_path: str, results: dict):
    """Save analysis report."""
    os.makedirs(output_path, exist_ok=True)
    report_path = Path(output_path) / 'report.md'
    content = getattr(results.get('messages', [])[-1], 'content', "No report generated.")
    report_path.write_text(content)
    log_print(f"Report saved to {report_path}")

def display_pipeline(pipeline_path: str):
    """Display pipeline if exists."""
    if Path(pipeline_path).exists():
        log_print("Loading and displaying pipeline...")
        model = Fedot(problem="classification")
        model.load(pipeline_path)
        model.current_pipeline.show()
        logging.info(graph_structure(model.current_pipeline))

def update_solution_timeout(solution_path: Path, timeout: int = 60):
    """Update timeout in solution file."""
    if not solution_path.exists():
        return log_print(f"Solution file {solution_path} does not exist.")
    
    log_print(f"Updating solution timeout to {timeout} minutes...")
    code = solution_path.read_text()
    pattern = r'(timeout\s*=\s*)([0-9]*\.?[0-9]+)'
    updated_code = re.sub(pattern, rf'\g<1>{timeout}', code)
    solution_path.write_text(updated_code)
    log_print(f"Updated {solution_path}")

def execute_solution(solution_path: Path, output_path: str):
    """Execute solution script."""
    if not solution_path.exists():
        return log_print(f"Solution file {solution_path} does not exist.")
    
    log_print("Executing solution...")
    process = subprocess.run(
        [sys.executable, str(solution_path)],
        cwd=output_path,
        capture_output=False,
        text=True
    )
    log_print(f"Process finished with return code: {process.returncode}")

def submit_solution(competition_name: str, submission_file: str):
    """Submit solution to Kaggle."""
    if not Path(submission_file).exists():
        return log_print(f"Submission file {submission_file} does not exist.")
    
    log_print(f"Submitting to Kaggle competition '{competition_name}'...")
    submit_to_kaggle(competition_name=competition_name, submission_file=submission_file)

async def run_fedot_ai_competition(
    dataset_path: str,
    output_path: str, 
    competition_name: str,
    description_file: str = "overview.txt",
    model: str = "github/gpt-4o-mini",
    log_file: str = None
):
    """Run FedotAI on Kaggle competition."""
    setup_logging(log_file)
    
    # Download data and setup workspace
    log_print(f"Downloading {competition_name} data...")
    download_from_kaggle(competition_name=competition_name, save_path=dataset_path)
    
    description = Path(description_file).read_text()
    log_print(f"Description preview:\n{description[:100]}...")
    
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Run FedotAI analysis
    log_print("=" * 40 + " FEDOT.LLM START " + "=" * 40)
    fedot_ai = FedotAI(
        task_path=dataset_path,
        inference=AIInference(model=model),
        workspace=output_path,
        handlers=JupyterOutput().subscribe
    )
    
    log_print("Running FedotAI analysis...")
    results = await fedot_ai.ainvoke(message=description)
    save_report(output_path, results)
    
    # Process results
    pipeline_path = Path(output_path) / 'pipeline'
    solution_path = Path(output_path) / "solution.py"
    submission_file = Path(output_path) / "submission.csv"
    
    # Initial submission
    display_pipeline(str(pipeline_path))
    submit_solution(competition_name, str(submission_file))
    
    # Extended timeout execution
    log_print("=" * 35 + " EXTENDED TIMEOUT (60 min) " + "=" * 35)
    update_solution_timeout(solution_path, 60)
    execute_solution(solution_path, output_path)
    display_pipeline(str(pipeline_path))
    submit_solution(competition_name, str(submission_file))
    
    return output_path

if __name__ == "__main__":
    import asyncio
    
    competition_name = "spaceship-titanic"
    dataset_path = f"examples/{competition_name}"
    output_path = f"examples/{competition_name}/output"
    
    result_path = asyncio.run(run_fedot_ai_competition(
        dataset_path=dataset_path,
        output_path=output_path,
        competition_name=competition_name,
        log_file=f"{competition_name}.log"
    ))
    
    print(f"Competition analysis completed. Results: {result_path}")