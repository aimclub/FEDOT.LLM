# FEDOT.LLM

<p align="center">
  <img src="/docs/fedot-llm.png" width="600" title="Fedot.LLM logo">
</p>


FEDOT.LLM is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.

## Installation

1. FEDOT.LLM is only available via github now. To install, clone the repository:
   ```
   git clone https://github.com/ITMO-NSS-team/FEDOT.LLM.git
   cd FEDOT.LLM
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Use

FEDOT.LLM provides a high-level API with simple interface through FedotAI class. It can be used to start the whole pipeline of LLM-powered dataset analysis and making predictions using FEDOT.

To use the API, follow these steps:

1. Import FedotAI class
   ```
   from fedot_llm.main import FedotAI
   ```

2. Initialize the FedotAI object. The required parameters are the following: 
   * `dataset` path to folder containing dataset files or pre-loaded dataset object
   * `model` chat model to use (currently ollama and custom request based models are supported)
   * `output` output type ('jupyter' for live updated status report and 'debug' for a feed of all langchain events)

   To acquire predictions, use the `predict` method with a string description of the dataset and associated task in an arbitrary form.
   ```
   fedot_ai =  FedotAI(
         dataset=dataset_path,
         model=init_chat_model(
                           model="llama3.1",
                           model_provider='ollama'),
         output='jupyter')
   predictions = await fedot_ai.predict(description)
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