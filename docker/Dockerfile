FROM python:3.10-slim

# Arguments for the new user
ARG NB_USER=fedotllm
ARG NB_UID=1000

# Set the environment variables and create a new user
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME="/home/${NB_USER}"

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Install the required packages
WORKDIR ${HOME}

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY . ${HOME}

# Switch to new user and start the Jupyter notebook
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}


ENTRYPOINT ["python3", "-m"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]