from pathlib import Path
import sys
import shutil
import asyncio

import litellm

from fedotllm.main import FedotAI
from fedotllm.handlers import JupyterOutput
from fedotllm.log import logger

async def run_experiment(dataset_path, output_path, description):
    logger.info("=" * 60)
    logger.info("EXPERIMENT STARTING")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Description: {description}")
    logger.info("=" * 60)

    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    description = ""

    fedot_ai = FedotAI(
            task_path=dataset_path,
            workspace=output_path,
            handlers=JupyterOutput().subscribe
        )

    try:
        async for _ in fedot_ai.ask(message=description):
            continue

        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"EXPERIMENT FAILED: {e}")
        logger.error("=" * 60)
        raise

if __name__ == "__main__":
    dataset_path = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/ml2b/actuarial-loss-prediction"
    output_path = Path(dataset_path) / "output"
    description = 'Develop a predictive model to estimate ultimate incurred claim costs for insurance policies. The dataset includes information such as age, gender, marital status, wages, and textual descriptions of claims. '

    asyncio.run(run_experiment(dataset_path, output_path, description))