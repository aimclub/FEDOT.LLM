# FEDOT.DATA-ANALYST

<p align="center">
  <img src="./docs/FEDOT-DATA-ANALYST-logo.svg" width="600" alt="Logo">
</p>

[![Acknowledgement ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)
[![Acknowledgement NCCR](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/NCCR_badge.svg)](https://actcognitive.org/)
[![Mirror](https://img.shields.io/badge/mirror-GitLab-orange)](https://gitlab.actcognitive.org/itmo-nccr-code/fedot-llm)


FEDOT.DATA-ANALYST is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.


## 💾 Installation

1. Install uv (A fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/AaLexUser/Fedot-assistant.git
git checkout data_analyst
cd Fedot-assistant
```

3. Create a new virtual environment and activate it:

```bash
uv venv --python 3.10
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```

4. Install dependencies:

```bash
uv sync
```

## How to Use

```python
from fedotllm.agents.data_analyst.data_analyst import DataAnalystAgent
from fedotllm.utils.configs import load_config
from pathlib import Path
from datetime import datetime
config = load_config()
workspace = Path(f"user_data/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
data_analyst = DataAnalystAgent(config=config, session_id=f"test-{datetime.now().strftime('%Y%m%d_%H%M%S')}", workspace=workspace)
result = await data_analyst.create_graph().ainvoke({"problem_description": problem_description, "workspace": workspace}, config={
    "recursion_limit": 100
})
result
```

## Examples and demo

You can use the example notebook in the `examples/data_analyst/main.ipynb`.

Funding
=======

This research is financially supported by the Foundation for
National Technology Initiative's Projects Support as a part of the roadmap
implementation for the development of the high-tech field of
Artificial Intelligence for the period up to 2030 (agreement 70-2021-00187)


