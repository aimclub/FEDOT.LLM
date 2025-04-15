import asyncio
import json
import logging
from typing import TYPE_CHECKING, Dict, List, Union

from langgraph.graph import END
from langgraph.types import Command

from fedotllm.agents.data_analyst.schema import Message
from fedotllm.agents.data_analyst.state import DataAnalystAgentState
from fedotllm.prompting import prompts
from fedotllm.tools.base import Observation
from fedotllm.tools.finish import FinishTool
from fedotllm.tools.planning import PlanningTool
from litellm.types.utils import ChatCompletionMessageToolCall

if TYPE_CHECKING:
    from fedotllm.agents.data_analyst.data_analyst import DataAnalystAgent

logger = logging.getLogger(__name__)


async def plan(
    self: "DataAnalystAgent", state: DataAnalystAgentState
) -> Union[Dict[str, List[str]], Command]:
    problem_description = state["problem_description"]
    self.memory.add_message(Message.system_message(content=problem_description))
    self.memory.add_message(
        Message.assistant_message(
            content=prompts.data_analyst.planner_prompt(
                self.library_functions, str(state["workspace"])
            )
        )
    )
    for attempt in range(3):
        try:
            # Use a more robust approach to handle the response
            response = await self.llm.aquery(
                messages=self.memory.to_dict_list(),
                tools=self.available_tools.to_params(
                    include=[PlanningTool, FinishTool]
                ),
                tool_choice="required",
            )

            # Check if the response is a tuple with content and tool_calls
            if isinstance(response, tuple) and len(response) == 2:
                content, tool_calls = response
            else:
                # Fall back to a default if the response format is unexpected
                logger.warning(f"Unexpected response format: {response}")
                content = str(response) if response else ""
                tool_calls = []

            self.memory.add_message(
                Message.assistant_message(content=content, tool_calls=tool_calls)
            )
            if len(tool_calls) == 0:
                self.memory.add_message(
                    Message.user_message(
                        content="You didn't provide any tool calls. You should provide at least one tool call."
                    )
                )
                continue
            elif tool_calls[0].function.name == "finish":
                return Command(
                    goto=END,
                )
            else:
                return Command(
                    goto="act",
                    update={"tool_calls": tool_calls},
                )

        except Exception as e:
            print(f"Error in plan_node: {e}, attempt {attempt + 1}")
            continue

    return Command(goto=END)


async def act(
    self: "DataAnalystAgent", state: DataAnalystAgentState
) -> Union[Dict[str, List[str]], Command]:
    tool_calls = state["tool_calls"]
    if tool_calls:
        for tool_call in tool_calls:
            result = await execute_tool(self, tool_call)

            logger.info(
                f"🎯 Tool '{tool_call.function.name}' completed its mission! Result: {result.message}"
            )
            # TODO: Add image to the tool message
            tool_msg = Message.tool_message(
                content=result.message,
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
            )
            self.memory.add_message(tool_msg)
    return Command(goto="think")


async def think(
    self: "DataAnalystAgent", state: DataAnalystAgentState
) -> Union[Dict[str, List[str]], Command]:
    self.memory.add_message(
        Message.assistant_message(
            content=prompts.data_analyst.think_prompt(
                self.library_functions,
                state["workspace"],
            )
        )
    )
    memory_list = self.memory.to_dict_list()
    for _ in range(3):
        try:
            # Use a more robust approach to handle the response
            response = await self.llm.aquery(
                messages=memory_list,
                tool_choice="required",
                tools=self.available_tools.to_params(),
            )

            # Check if the response is a tuple with content and tool_calls
            if isinstance(response, tuple) and len(response) == 2:
                content, tool_calls = response
            else:
                # Fall back to a default if the response format is unexpected
                logger.warning(f"Unexpected response format: {response}")
                content = str(response) if response else ""
                tool_calls = []

            self.memory.add_message(
                Message.assistant_message(content=content, tool_calls=tool_calls)
            )
            if len(tool_calls) == 0:
                self.memory.add_message(
                    Message.user_message(
                        content="You didn't provide any tool calls. You should provide at least one tool call."
                    )
                )
                continue
            elif tool_calls[0].function.name == "finish":
                return Command(
                    goto=END,
                )

            return Command(goto="act", update={"tool_calls": tool_calls})
        except Exception as e:
            print(f"Error in think_node: {e}")
            continue

    return Command(goto=END)


async def execute_tool(
    self: "DataAnalystAgent", tool: ChatCompletionMessageToolCall
) -> Observation:
    if not tool or not tool.function or not tool.function.name:
        return Observation(is_success=False, message="Error: Invalid command format")
    name = tool.function.name
    try:
        # Parse arguments
        args = json.loads(tool.function.arguments or "{}")

        # Execute the tool
        logger.info(f"🔧 Activating tool: '{name}'...")
        result = await self.available_tools.execute(name=name, tool_input=args)
        return result
    except json.JSONDecodeError:
        error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
        logger.error(
            f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{tool.function.arguments}"
        )
        return Observation(is_success=False, message=f"Error: {error_msg}")
    except Exception as e:
        error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
        logger.exception(error_msg)
        return Observation(is_success=False, message=f"Error: {error_msg}")
