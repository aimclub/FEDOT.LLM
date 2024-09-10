# FEDOT.LLM

FEDOT.LLM is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and model building processes.

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/ITMO-NSS-team/FEDOT.LLM.git
   cd FEDOT.LLM
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   You can use the example notebooks in the `examples/by_datasets/` directory to get started. For instance, to run the [Health_Insurance](datasets/Health_Insurance) dataset example:

   ```
   jupyter notebook examples/by_datasets/health_insurance.ipynb
   ```

4. In the notebook, you can use the FedotAI class to analyze datasets and make predictions.

5. You can also use the Streamlit web interface for a more interactive experience. To run it:
    ```
    streamlit run streamlit-app.py
    ```
    For more information how to setup and run Streamlit app see [`STREAMLIT_README.md`](STREAMLIT_README.md).

## Development

If you want to contribute or set up a development environment, you can use the provided dev container.

This will set up a fully-featured development environment in a container, either in GitHub Codespaces or using VS Code's Dev Containers extension.

For more information see [`.devcontainer/README.md`](.devcontainer/README.md).