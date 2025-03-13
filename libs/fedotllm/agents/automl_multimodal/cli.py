import argparse

from fedotllm.agents.automl_multimodal.automl import AutoMLMultimodalAgent
from fedotllm.data.loaders import PathDatasetLoader
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger, setup_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Automl Agent")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file")
    parser.add_argument("--problem", type=str, help="Problem description")
    args = parser.parse_args()
    setup_logger()

    dataset = PathDatasetLoader.load(args.dataset)
    automl = AutoMLMultimodalAgent(inference=AIInference(), dataset=dataset).create_graph().invoke({
        "description": args.problem
    })

    #TODO: Actual CLI launch

if __name__ == "__main__":
    main()
