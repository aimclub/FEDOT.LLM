DC = docker compose
EXEC = docker exec -it
APP_FILE = docker/docker-compose.yml
APP_CONTAINER = app
ENV = --env-file .env
SH = /bin/bash

.PHONY: build
build:
	${DC} ${ENV} -f ${APP_FILE} up -w --build

.PHONY: up
up:
	${DC} ${ENV} -f ${APP_FILE} up -w

.PHONY: down
down:
	${DC} -f ${APP_FILE} down

.PHONY: appsh
app-sh:
	${EXEC} -it ${APP_CONTAINER} ${SH}
app-py:
	${EXEC} -it ${APP_CONTAINER} poetry shell

streamlit-st:
	python -m streamlit run libs/fedotllm/streamlit-app.py --server.port=8080 --server.address=0.0.0.0