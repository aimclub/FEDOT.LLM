from typing import Dict, List, Literal, Optional

from fedotllm.tools.base import BaseTool, Observation

_PLANNING_TOOL_DESCRIPTION = """
A planning tool that allows the agent to create and manage plans for solving complex tasks.
The tool provides functionality for creating plans, updating plan steps, and tracking progress.
"""


class PlanningTool(BaseTool):
    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The command to execute. Available commands: create, update, mark_step.",
                "enum": [
                    "create",
                    "update",
                    "mark_step",
                    "get",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "Unique identifier for the plan. Required for create, update, set_active, and delete commands. Optional for get and mark_step (uses active plan if not specified).",
                "type": "string",
            },
            "title": {
                "description": "Title for the plan. Required for create command, optional for update command.",
                "type": "string",
            },
            "steps": {
                "description": "List of plan steps. Required for create command, optional for update command.",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "Index of the step to update (0-based). Required for mark_step command.",
                "type": "integer",
            },
            "step_status": {
                "description": "Status to set for a step. Used with mark_step command.",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    plan: dict = {}

    async def execute(
        self,
        *,
        command: Literal["create", "update", "mark_step", "get"],
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ) -> Observation:
        """
        Execute the planning tool with the given command and parameters.

        Parameters:
        - command: The operation to perform
        - plan_id: Unique identifier for the plan
        - title: Title for the plan (used with create command)
        - steps: List of steps for the plan (used with create command)
        - step_index: Index of the step to update (used with mark_step command)
        - step_status: Status to set for a step (used with mark_step command)
        - step_notes: Additional notes for a step (used with mark_step command)
        """
        if command == "create":
            return self._create_plan(title, steps)
        elif command == "update":
            return self._update_plan(title, steps)
        elif command == "mark_step":
            return self._mark_step(step_index, step_status, step_notes)
        elif command == "get":
            return self.get_plan()
        else:
            return Observation(
                is_success=False,
                message=f"Unrecognized command: {command}. Allowed commands are: create, update, mark_step",
            )

    def _create_plan(
        self, title: Optional[str], steps: Optional[List[str]]
    ) -> Observation:
        """Create a new plan with the given ID, title, and steps."""

        if not title:
            return Observation(
                is_success=False,
                message="Parameter `title` is required for command: create",
            )

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            return Observation(
                is_success=False,
                message="Parameter `steps` must be a non-empty list of strings for command: create",
            )

        # Create a new plan with initialized step statuses
        plan = {
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
        }

        self.plan = plan

        return Observation(
            is_success=True,
            message=f"Plan created successfully\n\n{self._format_plan(plan)}",
        )

    def _update_plan(
        self, title: Optional[str], steps: Optional[List[str]]
    ) -> Observation:
        """Update an existing plan with new title or steps."""

        if title:
            self.plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                return Observation(
                    is_success=False,
                    message="Parameter `steps` must be a list of strings for command: update",
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = self.plan["steps"]
            old_statuses = self.plan["step_statuses"]

            # Create new step statuses and notes
            new_statuses = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                else:
                    new_statuses.append("not_started")

            self.plan["steps"] = steps
            self.plan["step_statuses"] = new_statuses

        return Observation(
            is_success=True,
            message=f"Plan updated successfully\n\n{self._format_plan(self.plan)}",
        )

    def _mark_step(
        self,
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> Observation:
        """Mark a step with a specific status and optional notes."""

        if step_index is None:
            return Observation(
                is_success=False,
                message="Parameter `step_index` is required for command: mark_step",
            )

        if step_index < 0 or step_index >= len(self.plan["steps"]):
            return Observation(
                is_success=False,
                message=f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(self.plan['steps']) - 1}.",
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
        ]:
            return Observation(
                is_success=False,
                message=f"Invalid step_status: {step_status}. Valid statuses are: not_started, in_progress, completed",
            )

        if step_status:
            self.plan["step_statuses"][step_index] = step_status

        return Observation(
            is_success=True,
            message=f"Step {step_index} updated in plan.\n\n{self._format_plan(self.plan)}",
        )

    def get_plan(self) -> Observation:
        return Observation(
            is_success=True,
            message=self._format_plan(self.plan),
        )

    def _format_plan(self, plan: Dict) -> str:
        """Format a plan for display."""
        output = f"Plan: {plan['title']}\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {not_started} not started\n\n"
        output += "Steps:\n"

        # Add each step with its status and notes
        for i, (step, status) in enumerate(zip(plan["steps"], plan["step_statuses"])):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
        return output
