ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/devcontainers/python:0-${VARIANT} AS fedot_llm-dev-base

USER vscode

# Define the directory of python virtual environment
ARG PYTHON_VIRTUALENV_HOME=/home/vscode/fedot_llm-py-env

# Create a Python virtual environment for the project
RUN python3 -m venv ${PYTHON_VIRTUALENV_HOME} && \
    $PYTHON_VIRTUALENV_HOME/bin/pip install --upgrade pip

ENV PATH="$PYTHON_VIRTUALENV_HOME/bin:$PATH" \
    VIRTUAL_ENV=$PYTHON_VIRTUALENV_HOME

# Set the working directory for the app
WORKDIR /workspaces/fedot_llm


# Use a multi-stage build to install dependencies
FROM fedot_llm-dev-base AS fedot_llm-dev-dependencies

ARG PYTHON_VIRTUALENV_HOME=/home/vscode/fedot_llm-py-env

# Copy only the dependency files for installation
COPY requirements.txt ./

# Copy the fedot_llm lib
COPY fedot_llm .

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt