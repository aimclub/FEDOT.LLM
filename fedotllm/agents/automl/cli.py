from agents.automl.llm.inference import AIInference
import argparse
from agents.automl.automl import AutoMLAgent
from agents.automl.data.loaders import PathDatasetLoader
from agents.automl.eval.local_exec import ProgramStatus
from log import get_logger, setup_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Automl Agent")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file")
    parser.add_argument("--problem", type=str, help="Problem description")
    args = parser.parse_args()
    setup_logger()

    dataset = PathDatasetLoader.load(args.dataset)
    automl = AutoMLAgent(inference=AIInference(), dataset=dataset).create_graph().invoke({
        "description": args.problem
    })

    if automl['solutions'][-1]['exec_result'].program_status == ProgramStatus.kSuccess:
        logger.info("Solution found")
        logger.debug(f"Solutions:\n{automl['solutions']}")
    else:
        logger.error("No solution found")


if __name__ == "__main__":
    main()
