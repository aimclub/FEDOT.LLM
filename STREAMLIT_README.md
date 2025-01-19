# Streamlit and Fedot.LLM

To run web interface for FEDOT.LLM follow these steps:

### Step 1: Clone the repository

Run the following command in your terminal to clone the repository:

```bash
git clone https://github.com/aimclub/FEDOT.LLM.git 
```
### Step 2: Set up a virtual environment

Navigate to the project directory and create a virtual environment:

``` bash
cd FEDOT.LLM
python3.10 -m venv .venv
```

Activate the virtual environment:
* On Windows
``` bash
.venv\Scripts\activate
```
* On macOS / Linux
``` bash
source .venv/bin/activate
```

### Step 3: Install dependencies

``` bash
pip install -e .
```

This will install all the necessary libraries, including Streamlit, Fedot-LLM, and others.  

**Note: Before running the application make sure to set the OPENAI_TOKEN environment variable**

### Step 4: Run the Streamlit application

Run the following command to start the Streamlit application:

``` bash
streamlit run libs/fedotllm/streamlit-app.py
```

This will launch the Streamlit app in your default web browser.
