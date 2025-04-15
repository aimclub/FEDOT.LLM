from typing import Any, Dict, List, Type

from fedotllm.tools.base import BaseTool, Observation


class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(
        self, exclude: List[Type[BaseTool]] = [], include: List[Type[BaseTool]] = []
    ) -> List[Dict[str, Any]]:
        return [
            tool.to_param()
            for tool in self.tools
            if not isinstance(tool, tuple(exclude))
            and (not include or isinstance(tool, tuple(include)))
        ]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = {}
    ) -> Observation:
        tool = self.tool_map.get(name)
        if not tool:
            return Observation(message=f"Tool {name} is invalid", is_success=False)
        try:
            result = await tool(**tool_input)
            return result
        except Exception as e:
            return Observation(message=str(e), is_success=False)

    async def execute_all(self) -> List[Observation]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except Exception as e:
                results.append(Observation(message=str(e), is_success=False))
        return results

    def get_tool(self, name: str) -> BaseTool | None:
        return self.tool_map.get(name, None)

    def add_tool(self, tool: BaseTool):
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        for tool in tools:
            self.add_tool(tool)
        return self
