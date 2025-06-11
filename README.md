# FEDOT.LLM

<p align="center">
  <img src="./docs/fedot-llm.png" width="600" title="Fedot.LLM logo">
</p>

[![Acknowledgement ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)
[![Acknowledgement NCCR](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/NCCR_badge.svg)](https://actcognitive.org/)
[![Mirror](https://img.shields.io/badge/mirror-GitLab-orange)](https://gitlab.actcognitive.org/itmo-nccr-code/fedot-llm)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/aimclub/FEDOT.LLM)

FEDOT.LLM is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.

## ⚙️ Installation and Setup


### 📦 Basic Installation

We offer two installation methods to suit your preferences:


#### 🚀 Method 1: Using uv (Recommended)

<details>
<summary><b>📋 Step-by-step installation with uv</b></summary>

**Step 1: Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2: Clone the repository**
```bash
git clone https://github.com/aimclub/FEDOT.LLM.git
cd FEDOT.LLM
```

**Step 3: Create and activate virtual environment**
```bash
uv venv --python 3.10
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```

**Step 4: Install dependencies**
```bash
uv sync
```

</details>

#### 🐍 Method 2: Using conda

<details>
<summary><b>📋 Step-by-step installation with conda</b></summary>

**Step 1: Create conda environment**
```bash
conda create -n FedotLLM python=3.10
conda activate FedotLLM
```

**Step 2: Clone the repository**
```bash
git clone https://github.com/aimclub/FEDOT.LLM.git
cd FEDOT.LLM
```

**Step 3: Install dependencies**
```bash
pip install -e .
```

</details>

### 🔧 Environment Configuration

FEDOT.LLM requires API keys to access external services. Configure them through environment variables for seamless operation.

#### Option 1: Create `.env` file (Recommended)

Create a `.env` file in the project root:

```bash
# Required API Keys
FEDOTLLM_LLM_API_KEY=your_llm_api_key_here
FEDOTLLM_EMBEDDINGS_API_KEY=your_embeddings_api_key_here

# Optional: For tracing LLM calls with Langfuse
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
```

#### Option 2: Export directly

```bash
export FEDOTLLM_LLM_API_KEY=your_llm_api_key_here
export FEDOTLLM_EMBEDDINGS_API_KEY=your_embeddings_api_key_here

# Optional: For tracing LLM calls with Langfuse
export LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
export LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
```

<div align="center">

**🎉 Congratulations! You're ready to explore FEDOT.LLM**

</div>

## How to Use

FEDOT.LLM provides a high-level API with simple interface through FedotAI class. It can be used to start the whole pipeline of LLM-powered dataset analysis and making predictions using FEDOT.

To use the API, follow these steps:

1. Import FedotAI class
   ```
   from fedotllm.main import FedotAI
   ```

2. Initialize the FedotAI object. The following parameters are required:

* The `task_path` parameter specifies the directory path where the competition files are located.
* The `inference` parameter chat model to be utilized. A comprehensive list of supported models and providers can be accessed via the litellm official documentation at [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).
* The `handlers` parameter is a list of output handlers to be utilized. It is possible to develop custom output handlers or utilize existing ones. For example, `JupyterOutput` includes handlers specifically designed for Jupyter notebooks. To subscribe to all available handlers, use the `subscribe` attribute.

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
inference = AIInference(model="openai/gpt-4o", api_key=os.getenv('FEDOTLLM_LLM_API_KEY'))

# Set the path to the dataset
# Load the dataset using PathDatasetLoader
dataset_path = Path('datasets') / 'Health_Insurance'

# Define the task description for the model
msg="""Create a model that perform this task:
Our client is an insurance company that has provided health insurance to its customers.
They are interested in whether the policyholders (customers) from last year
will also be interested in the car insurance provided by the company."""

# Initialize FedotAI with the dataset, language model, and output handlers
fedot_ai = FedotAI(
        task_path=dataset_path,
        inference=inference,
        workspace=output_path,
        handlers=JupyterOutput().subscribe
    )

# Asynchronously process the task using FedotAI
# The loop continues until the task is completed
async for _ in fedot_ai.ask(message=msg):
    continue
```

## Examples and demo

You can also use the Streamlit web interface for a more interactive experience. To run it:

```zsh
uv run python -m streamlit run fedotllm/web/streamlit-app.py
```

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