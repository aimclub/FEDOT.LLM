# FEDOT.LLM

<p align="center">
  <img src="/docs/fedot-llm.png" width="600" title="Fedot.LLM logo">
</p>

[![Acknowledgement ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)
[![Acknowledgement NCCR](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/NCCR_badge.svg)](https://actcognitive.org/)
[![Mirror](https://img.shields.io/badge/mirror-GitLab-orange)](https://gitlab.actcognitive.org/itmo-nccr-code/fedot-llm)


FEDOT.LLM is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.

## Installation

1. FEDOT.LLM is only available via github now:
   ```
   conda create -n FedotLLM python=3.10
   pip install git+https://github.com/aimclub/FEDOT.LLM.git
   ```

## How to Use

FEDOT.LLM provides a high-level API with simple interface through FedotAI class. It can be used to start the whole pipeline of LLM-powered dataset analysis and making predictions using FEDOT.

To use the API, follow these steps:

1. Import FedotAI class
   ```
   from fedotllm.main import FedotAI
   ```

2. Initialize the FedotAI object. The required parameters are the following: 

* The `dataset` is a native `fedot_llm.data.data.Dataset` object that contains the dataset files. It can be initialized using specific loaders, such as the `PathDatasetLoader`.

* The `model` is the chat model you want to use. You can use any chat model class from the `langchain` library. However, for the best experience, we recommend using models like gpt4o-mini or higher.

* `handlers` is a list of output handlers to use. You can create your own output handler or use the pre-existing ones. For instance, `JupyterOutput` contains handlers for Jupyter notebooks. You can subscribe to all of them using `JupyterOutput().subscribe`.

To acquire predictions, use the `ask` method with a string description of the dataset and associated task in an arbitrary form.
```python
# Import necessary modules and classes
import os
from pathlib import Path

from fedotllm.data.loaders import PathDatasetLoader
from fedotllm.llm.inference import AIInference
from fedotllm.main import FedotAI
from fedotllm.output.jupyter import JupyterOutput

# Initialize the LLM model
inference = AIInference(model="gpt-4o-mini", api_key=os.getenv('OPENAI_TOKEN'), base_url='https://models.inference.ai.azure.com')

# Set the path to the dataset
# Load the dataset using PathDatasetLoader
dataset_path = Path('datasets') / 'Health_Insurance'
dataset = PathDatasetLoader.load(dataset_path)

# Define the task description for the model
msg="""Create a model that perform this task:
Our client is an insurance company that has provided health insurance to its customers.
They are interested in whether the policyholders (customers) from last year
will also be interested in the car insurance provided by the company."""

# Initialize FedotAI with the dataset, language model, and output handlers
fedot_ai = FedotAI(dataset=dataset,
                   inference=inference,
                   handlers=JupyterOutput().subscribe)

# Asynchronously process the task using FedotAI
# The loop continues until the task is completed
async for _ in fedot_ai.ask(message=msg):
    continue
```

## Examples and demo

You can use the example notebooks in the `examples/by_datasets/` directory to get started. For instance, to run the [Health_Insurance](datasets/Health_Insurance) dataset example:
   ```
   jupyter notebook examples/by_datasets/health_insurance.ipynb
   ```

You can also use the Streamlit web interface for a more interactive experience. To run it:
   ```
   streamlit run streamlit-app.py
   ```

For more information on how to setup and run Streamlit app see [`STREAMLIT_README.md`](STREAMLIT_README.md).

## Development

If you want to contribute or set up a development environment, you can use the provided dev container.

This will set up a fully-featured development environment in a container, either in GitHub Codespaces or using VS Code's Dev Containers extension.

For more information see [`.devcontainer/README.md`](.devcontainer/README.md).

Funding
=======

This research is financially supported by the Foundation for
National Technology Initiative's Projects Support as a part of the roadmap
implementation for the development of the high-tech field of
Artificial Intelligence for the period up to 2030 (agreement 70-2021-00187)


