from typing import TYPE_CHECKING, Any, Dict, Literal

from pydantic import Field

from fedotllm.environments.jupyter import JupyterExecutor
from fedotllm.tools.base import BaseTool, Observation

if TYPE_CHECKING:
    from fedotllm.environments.jupyter import JupyterExecutor


class JupyterTool(BaseTool):
    executor: JupyterExecutor
    name: str = Field(default="jupyter")
    description: str = Field(
        default="Use this tool to add code cell to the notebook and execute it. Tool maintains kernel state between calls so variables and imports persist across executions."
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code or markdown content to run.",
                },
                "language": {
                    "type": "string",
                    "description": "The language of the code. Either 'python' or 'markdown'.",
                    "enum": ["python", "markdown"],
                },
            },
        }
    )

    async def execute(
        self, code: str, language: Literal["python", "markdown"] = "python"
    ) -> Observation:
        return await self.executor.run(code, language)
