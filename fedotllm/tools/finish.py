from fedotllm.tools.base import BaseTool, Observation


class FinishTool(BaseTool):
    name: str = "finish"
    description: str = "Finish the execution of the plan"
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> Observation:
        return Observation(
            is_success=True,
            message="Successfully finished the execution of the plan. You can reply to the user now.",
        )
