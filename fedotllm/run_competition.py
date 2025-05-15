import os
import sys
import argparse
import asyncio

from dotenv import load_dotenv

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from fedotllm.data import Dataset
from fedotllm.main import FedotAI
from fedotllm.llm.inference import AIInference
from fedotllm.output.jupyter import JupyterOutput
from fedotllm.settings.config_loader import get_settings
from fedotllm.log import get_logger

logger = get_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a competition model with configurable inputs.")

    parser.add_argument(
        "--competition",
        type=str,
        default="titanic",
        help="Name of a single competition (default: 'titanic'). Can be a name or a path."
    )

    parser.add_argument(
        "--competitions_file",
        type=str,
        help="Path to a .txt file containing competition names, or just a filename (without extension) assumed to be under the default path. Optional."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the model to use (default: 'gpt-4o-mini')."
    )

    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for the API inference."
    )

    parser.add_argument(
        "--runs_folder",
        type=str,
        default="_experiments",
        help="Optional path for outputs. Defaults to '_experiments'"
    )

    parser.add_argument(
        "--base_url",
        type=str,
        help="Optional base URL for API calls. Loads from config.toml if not specified"
    )

    return parser.parse_args()

def process_arguments(args):
    # Base path
    repo_path = Path(__file__).resolve().parent.parent
    base_competition_path = repo_path / 'competition'

    competitions = []

    if args.competitions_file:
        # Determine the actual file path
        file_path = Path(args.competitions_file)
        print(file_path)
        if not file_path.is_absolute():
            file_path = base_competition_path / (file_path.stem + ".txt")

        if not file_path.exists():
            raise FileNotFoundError(f"Competitions file not found: {file_path}")

        with open(file_path, 'r') as f:
            competitions = [line.strip() for line in f if line.strip()]
        print(f"Loaded competitions from file: {competitions}")

    else:
        # Use single competition argument
        comp_path = Path(args.competition)
        if comp_path.is_absolute() or comp_path.exists():
            competition_full_path = comp_path
        else:
            competition_full_path = base_competition_path / args.competition
        competitions = [competition_full_path]
        print(f"Using single competition: {competition_full_path}")

    args.competitions = competitions
    args.base_url = args.base_url or get_settings().get("config.base_url", None)

    if not args.base_url:
        raise ValueError("no base_url specified in args or config.toml")

    return args

async def run_competiton(comp_path: Path, args):

    comp_name = comp_path.stem
    task_name = f'{comp_name}'
    
    overview_path = comp_path / 'overview.txt'
    with open(overview_path,"r") as f:
        overview = f.read()
        
    api_key=os.environ['OPENAI_TOKEN']

    fedot_ai = FedotAI(task_path=comp_path, 
                       inference=AIInference(
                           model=args.model,
                           base_url=args.base_url,
                           api_key=api_key,
                           tag = args.tag
                           ),
                    handlers=JupyterOutput().subscribe,
                    automl_only = True)
    async for _ in fedot_ai.ask(overview):
        continue
def _set_env(var: str):
    if not os.environ.get(var):
        print(f"No {var} in env")
if __name__ == "__main__":
    load_dotenv()
    _set_env("OPENAI_TOKEN")
    _set_env("VSEGPT_TOKEN")

    args = parse_arguments()
    config = process_arguments(args)
 
    for competition_path in args.competitions:
        logger.info(f"Launching competition {competition_path.stem} from {competition_path}")
        logger.info(get_settings().get("OPENAI_TOKEN"))
        asyncio.run(run_competiton(competition_path, args))
