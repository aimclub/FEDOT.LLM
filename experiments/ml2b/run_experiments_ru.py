from pathlib import Path
import sys
import shutil
import asyncio
import json

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

async def run_experiments_series():
    """Run a series of experiments based on competition_descriptions_ru.json"""
    # Load competition descriptions
    descriptions_path = Path(__file__).parent / "competition_descriptions.json"
    with open(descriptions_path, "r", encoding="utf-8") as f:
        competitions = json.load(f)

    # Filter to only run experiments from this list
    experiments_left = [
        #"biker-tour-recommendation",
        "actuarial-loss-prediction",
        #"she-hacks-2021",
    ]

    # Filter competitions to only those in experiments_left
    competitions = {comp_id: desc for comp_id, desc in competitions.items() if comp_id in experiments_left}

    ml2b_base = Path(__file__).parent / "ml2b_ru"

    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENTS SERIES - {len(competitions)} competitions")
    logger.info("=" * 80)

    successful = []
    failed = []

    for comp_id, description in competitions.items():
        dataset_path = ml2b_base / comp_id
        output_path = dataset_path / "output"

        # Check if dataset exists
        if not dataset_path.exists():
            logger.warning(f"Skipping {comp_id}: dataset path does not exist")
            failed.append((comp_id, "Dataset path not found"))
            continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Competition {len(successful) + len(failed) + 1}/{len(competitions)}: {comp_id}")
        logger.info(f"{'=' * 80}")

        try:
            # Run experiment with 25 minute timeout
            await asyncio.wait_for(
                run_experiment(str(dataset_path), str(output_path), description),
                timeout=25 * 60  # 25 minutes in seconds
            )
            successful.append(comp_id)
        except asyncio.TimeoutError:
            logger.error(f"Competition {comp_id} timed out after 25 minutes")
            failed.append((comp_id, "Timeout after 25 minutes"))
        except Exception as e:
            logger.error(f"Competition {comp_id} failed: {e}")
            failed.append((comp_id, str(e)))

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTS SERIES COMPLETED")
    logger.info(f"Successful: {len(successful)}/{len(competitions)}")
    logger.info(f"Failed: {len(failed)}/{len(competitions)}")
    logger.info("=" * 80)

    if successful:
        logger.info("\nSuccessful competitions:")
        for comp_id in successful:
            logger.info(f"  ✓ {comp_id}")

    if failed:
        logger.error("\nFailed competitions:")
        for comp_id, error in failed:
            logger.error(f"  ✗ {comp_id}: {error}")

if __name__ == "__main__":
    asyncio.run(run_experiments_series())