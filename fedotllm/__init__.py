import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from omegaconf import OmegaConf
from rich import print as rprint

from fedotllm.constants import DEFAULT_QUALITY, PRESETS
from fedotllm.main import FedotAI
from fedotllm.utils import load_config

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


@dataclass
class TimingContext:
    start_time: float
    total_time_limit: float

    @property
    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def time_remaining(self) -> float:
        return self.total_time_limit - self.time_elapsed


@contextmanager
def time_block(description: str, timer: TimingContext):
    """Context manager for timing code blocks and logging the duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logging.info(
            f"It took {duration:.2f} seconds {description}. "
            f"Time remaining: {timer.time_remaining:.2f}/{timer.total_time_limit:.2f}"
        )


def run_ui(
    task_path: Annotated[
        str, typer.Argument(help="Directory where task files are included")
    ],
    presets: Annotated[
        Optional[str],
        typer.Option("--presets", "-p", help="Presets"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path", "-c", help="Path to the configuration file (config.yaml)"
        ),
    ] = None,
    config_overrides: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config_overrides",
            "-o",
            help="Override config values. Format: key=value or key.nested=value. Can be used multiple times.",
        ),
    ] = None,
    workspace_path: Annotated[Optional[str], typer.Option(help="Workspace path")] = "",
):
    logging.info("Starting FedotLLM")

    if presets is None or presets not in PRESETS:
        logging.info(f"Presets is not provided or invalid: {presets}")
        presets = DEFAULT_QUALITY
        logging.info(f"Using default presets: {presets}")
    logging.info(f"Presets: {presets}")

    # Load config with all overrides
    try:
        config = load_config(presets, config_path, config_overrides)
        logging.info("Successfully loaded config")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    rprint("🤖 [bold red] Welcome to FEDOT.LLM [/bold red]")

    rprint("Will use task config:")
    rprint(OmegaConf.to_container(config))

    task_path = Path(task_path).resolve()
    assert task_path.is_dir(), (
        "Task path does not exist, please provide a valid directory."
    )
    rprint(f"Task path: {task_path}")

    fedot_ai = FedotAI(config=config, task_path=task_path, workspace=workspace_path)
    return fedot_ai


def run_fedotllm(
    task_path: Annotated[
        str, typer.Argument(help="Directory where task files are included")
    ],
    presets: Annotated[
        Optional[str],
        typer.Option("--presets", "-p", help="Presets"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path", "-c", help="Path to the configuration file (config.yaml)"
        ),
    ] = None,
    config_overrides: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config_overrides",
            "-o",
            help="Override config values. Format: key=value or key.nested=value. Can be used multiple times.",
        ),
    ] = None,
    workspace_path: Annotated[Optional[str], typer.Option(help="Workspace path")] = "",
):
    logging.info("Starting FedotLLM")

    if presets is None or presets not in PRESETS:
        logging.info(f"Presets is not provided or invalid: {presets}")
        presets = DEFAULT_QUALITY
        logging.info(f"Using default presets: {presets}")
    logging.info(f"Presets: {presets}")

    # Load config with all overrides
    try:
        config = load_config(presets, config_path, config_overrides)
        logging.info("Successfully loaded config")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    rprint("🤖 [bold red] Welcome to FEDOT.LLM [/bold red]")

    rprint("Will use task config:")
    rprint(OmegaConf.to_container(config))

    task_path = Path(task_path).resolve()
    assert task_path.is_dir(), (
        "Task path does not exist, please provide a valid directory."
    )
    rprint(f"Task path: {task_path}")

    fedot_ai = FedotAI(config=config, task_path=task_path, workspace=workspace_path)
    return fedot_ai


def main():
    app = typer.Typer()
    app.command()(run_fedotllm)
    app()


if __name__ == "__main__":
    main()
