# Dev container

This project includes a dev container, which lets you use a container as a full-featured dev environment.

You can use the dev container configuration in this folder to build and run the app without needing to install any of its tools locally! You can use it in [GitHub Codespaces](https://github.com/features/codespaces) or the [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/stas1f1/AutoML-LLM)

You may use the button above, or follow these steps to open this repo in a Codespace:
1. Click the **Code** drop-down menu at the top of repo.
1. Click on the **Codespaces** tab.
1. Click **Create codespace on master**.

For more info, check out the [GitHub documentation](https://docs.github.com/en/free-pro-team@latest/github/developing-online-with-codespaces/creating-a-codespace#creating-a-codespace).


## VS Code Dev Containers

Follow these steps to open this repo in a container using the VS Code Dev Containers extension:

1. If this is your first time using a development container, please ensure your system meets the pre-reqs (i.e. have Docker installed) in the [getting started steps](https://aka.ms/vscode-remote/containers/getting-started).

2. Open a locally cloned copy of the code:

   - Fork and Clone this repository to your local filesystem.
   - Press <kbd>F1</kbd> and select the **Dev Containers: Open Folder in Container...** command.
   - Select the cloned copy of this folder, wait for the container to start, and try things out!

You can learn more in the [Dev Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).

## Tips and tricks
* To connect to ollama, you should use http://ollama:11434 inside the container 
* If you'd like to review the contents of the image used in this dev container, you can check it out in the [devcontainers/images](https://github.com/devcontainers/images/tree/main/src/python) repo.

