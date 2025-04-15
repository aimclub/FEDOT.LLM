import asyncio
import atexit
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import nbformat
from nbclient.client import KernelClient, KernelManager
from nbformat import NotebookNode
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
    new_output,
    output_from_msg,
)
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from fedotllm.tools.base import Observation

"""
Jupyter Notebook integration module for FEDOT.LLM.

This module provides functionality for programmatically interacting with Jupyter notebooks,
including executing code, displaying rich outputs, and managing notebook cells and kernels.
It enables the execution of Python code and rendering of markdown content within notebooks,
handling various types of outputs including text, images, and error messages.

The main class, JupyterExecutor, provides a comprehensive interface for notebook operations,
while utility functions handle specific tasks like output formatting and display.
"""

logger = logging.getLogger(__name__)
INSTALL_KEEPLEN = 500

# Keep track of active executor instances for cleanup on exit
active_executors: list["JupyterExecutor"] = []


def shutdown_all_kernels():
    """Cleanup all active executors when the Python interpreter exits."""
    for executor in active_executors:
        try:
            # Create a new event loop for cleanup if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(executor.terminate())
            loop.close()
        except Exception as e:
            # Just log errors during cleanup, don't raise
            print(f"Error during executor cleanup: {str(e)}")


# Register the cleanup function to run at exit
atexit.register(shutdown_all_kernels)


class JupyterExecutor:
    """
    A class for executing code in Jupyter notebooks programmatically.

    This executor can handle both Python code and markdown cells, managing the kernel
    lifecycle and providing rich output display capabilities. It saves executed notebooks
    to the specified workspace directory.

    Attributes:
        nb (NotebookNode): The notebook object containing cells and metadata.
        workspace (Path): Directory path where notebooks and related files are stored.
        km (Optional[KernelManager]): The kernel manager for this notebook.
        kc (Optional[KernelClient]): The kernel client for this notebook.
        console (Console): Rich console for displaying outputs.
        interaction (Literal["ipython", None]): Indicates if running in IPython environment.
    """

    def __init__(self, workspace: Path, nb: NotebookNode = new_notebook()):
        """
        Initialize a new JupyterExecutor instance.

        Args:
            workspace (Path): Directory path where notebooks and related files are stored.
            nb (NotebookNode, optional): Existing notebook to use. Defaults to a new notebook.
        """
        self.nb = nb
        self.workspace = workspace
        # Initialize kernel manager and client as None
        self.km: Optional[KernelManager] = None  # Will store the kernel manager
        self.kc: Optional[KernelClient] = None  # Will store the kernel client
        self.console = Console()
        self.interaction: Literal["ipython", None] = (
            "ipython" if self.detect_ipython() else None
        )
        # Register this instance for cleanup on exit
        active_executors.append(self)

    def _display(
        self, code: str, language: Literal["python", "markdown"] = "python"
    ) -> None:
        """
        Display code or markdown content in the console.

        Args:
            code (str): The code or markdown content to display.
            language (Literal["python", "markdown"], optional): The language type. Defaults to "python".

        Raises:
            ValueError: If language is not "python" or "markdown".
        """
        if language == "python":
            self.console.print(
                Syntax(code, "python", theme="paraiso-dark", line_numbers=True)
            )
        elif language == "markdown":
            render_markdown(code)
        else:
            raise ValueError(f"Only support for python, markdown, but got {language}")

    async def init_kernel(self) -> None:
        """
        Initialize the kernel if needed.

        Creates a persistent kernel manager and client if they don't exist yet.
        """
        if self.km is None or self.kc is None:
            logger.debug("Creating new kernel manager and client")
            # Create a kernel manager
            from jupyter_client.manager import KernelManager

            self.km = KernelManager(kernel_name="python3")
            # Start the kernel
            self.km.start_kernel()

            # Create a kernel client and start channels
            self.kc = self.km.client()
            # Start all the channels
            self.kc.start_channels()

            # Wait for kernel to be ready
            self.kc.wait_for_ready()
            # Log the kernel ID for debugging
            if hasattr(self.km, "kernel_id"):
                kernel_id: str = self.km.kernel_id
                logger.debug(f"Kernel started with ID: {kernel_id}")
        else:
            logger.debug("Reusing existing kernel")

    async def terminate(self) -> None:
        """
        Terminate the running kernel and clean up resources.
        """
        if self.km is not None:
            try:
                logger.debug("Terminating kernel...")

                # Close the client channels first
                if self.kc is not None:
                    try:
                        self.kc.stop_channels()
                    except Exception as e:
                        logger.warning(f"Error stopping channels: {str(e)}")
                    self.kc = None

                # Then shutdown the kernel
                self.km.shutdown_kernel(now=True)
                self.km = None

                logger.debug("Kernel terminated")
            except Exception as e:
                logger.warning(f"Error during kernel termination: {str(e)}")
                # Reset references even if shutdown failed
                self.kc = None
                self.km = None

    async def reset(self) -> None:
        """
        Reset the kernel completely by terminating the current one and building a new one.

        This will create a fresh kernel and lose all previous variable context.
        """
        # Log that we're resetting the kernel
        logger.info("Resetting kernel...")

        # Terminate the current kernel
        await self.terminate()

        # Ensure we wait enough time for the kernel to fully shut down
        await asyncio.sleep(1.5)

        # Create a new kernel
        await self.init_kernel()

        logger.info("Kernel reset complete")

    def add_code_cell(self, code: str) -> None:
        """
        Add a new code cell to the notebook.

        Args:
            code (str): The Python code to add as a new cell.
        """
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str) -> None:
        """
        Add a new markdown cell to the notebook.

        Args:
            markdown (str): The markdown content to add as a new cell.
        """
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def display_image(
        self, image_base64: str, interaction_type: Literal["ipython", None]
    ) -> None:
        """
        Display a figure from base64-encoded image data.

        Args:
            image_base64 (str): The base64-encoded image data.
            interaction_type (Literal["ipython", None]): The type of execution environment.
                If "ipython", uses IPython's display system, otherwise uses PIL to show the image.
        """
        try:
            image_bytes = base64.b64decode(image_base64)
            if interaction_type == "ipython":
                from IPython.display import Image, display

                display(Image(data=image_bytes))
            else:
                # Only import PIL when needed to avoid unnecessary dependencies
                # import io
                # from PIL import Image as PILImage

                # image = PILImage.open(io.BytesIO(image_bytes))
                # Don't call show() as it opens external viewers which isn't ideal in most contexts
                # Just log that an image was detected
                logger.debug("Image detected in output (not displayed in console mode)")
        except Exception as e:
            logger.warning(f"Error displaying image: {str(e)}")
            # Continue execution even if image display fails

    def detect_ipython(self) -> bool:
        """
        Check if the code is running in an IPython environment.

        Returns:
            bool: True if running in an IPython environment, False otherwise.
        """
        try:
            # If running in Jupyter Notebook, __file__ variable does not exist
            from IPython import get_ipython

            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                return True
            else:
                return False
        except NameError:
            return False

    def parse_outputs(
        self, outputs: List[Dict[str, Any]], keep_len: int = 5000
    ) -> Tuple[bool, str, List[str]]:
        """
        Parse the outputs from a cell execution.

        Processes different types of outputs (display data, execute results, errors)
        and formats them appropriately. For display data with images, collects the images.

        Args:
            outputs (List[Dict[str, Any]]): List of output dictionaries from cell execution.
            keep_len (int, optional): Maximum length of text output to keep. Defaults to 5000.

        Returns:
            Tuple[bool, str, List[str]]: A tuple containing:
                - bool: Success status (True if no errors, False otherwise)
                - str: The formatted output text
                - List[str]: List of base64 encoded images
        """
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        images = []

        for i, output in enumerate(outputs):
            output_text = ""
            if output["output_type"] == "display_data":
                if "image/png" in output["data"]:
                    # Show the image in console if appropriate
                    self.display_image(output["data"]["image/png"], self.interaction)
                    # Also store it for returning
                    images.append(output["data"]["image/png"])
                else:
                    logger.info(
                        f"{i}th output['data'] from nbclient outputs dont have image/png, continue next output ..."
                    )
            elif output["output_type"] == "execute_result":
                output_text = output["data"]["text/plain"]
            elif output["output_type"] == "stream":
                # Handle stream output (like print statements)
                output_text = output.get("text", "")
            elif output["output_type"] == "error":
                output_text, is_success = "\n".join(output["traceback"]), False

            # handle coroutines that are not executed asynchronously
            if output_text.strip().startswith("<coroutine object"):
                output_text = "Executed code failed, you need use key word 'await' to run a async code."
                is_success = False

            output_text = strip_ansi_codes(output_text)
            if is_success:
                output_text = filter_log_lines(output_text)
            # The useful information of the exception is at the end,
            # the useful information of normal output is at the begining.
            if "<!DOCTYPE html>" not in output_text:
                output_text = (
                    output_text[:keep_len] if is_success else output_text[-keep_len:]
                )

            parsed_output.append(output_text)

        return is_success, ",".join([p for p in parsed_output if p]), images

    async def run_cell(
        self, cell: NotebookNode, cell_id: int
    ) -> Tuple[bool, str, List[str]]:
        """
        Execute a single notebook cell using our persistent kernel.

        Args:
            cell (NotebookNode): The cell to execute.
            cell_id (int): The index of the cell in the notebook.

        Returns:
            Tuple[bool, str, List[str]]: A tuple containing:
                - bool: Success status
                - str: Output text
                - List[str]: List of base64 encoded images
        """
        # Ensure we have a kernel
        await self.init_kernel()

        if self.kc is None:
            error_msg = "Kernel client not initialized"
            logger.error(error_msg)
            return False, error_msg, []

        kernel_client = cast(KernelClient, self.kc)

        try:
            # Execute the cell using our client directly
            msg_id = kernel_client.execute(cell.source)
            outputs = []

            # Process messages from the kernel
            while True:
                try:
                    msg = await kernel_client._async_get_iopub_msg(timeout=30)
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue

                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "status" and content["execution_state"] == "idle":
                        # Execution is complete when kernel becomes idle
                        break

                    if msg_type in (
                        "execute_result",
                        "display_data",
                        "stream",
                        "error",
                    ):
                        output = output_from_msg(msg)
                        outputs.append(output)

                except Exception as e:
                    logger.warning(f"Error getting message: {str(e)}")
                    break

            # Add outputs to the cell
            cell["outputs"] = outputs

            # Parse the outputs
            return self.parse_outputs(outputs)

        except Exception as e:
            logger.error(f"Error executing cell: {str(e)}")
            error_output = new_output(
                output_type="error",
                ename="ExecutionError",
                evalue=str(e),
                traceback=[str(e)],
            )
            cell["outputs"] = [error_output]
            return False, str(e), []

    async def run(
        self, code: str, language: Literal["python", "markdown"] = "python"
    ) -> Observation:
        """
        Run code in the notebook.

        Displays the code, executes it if it's Python code, or adds it as a markdown cell,
        and saves the notebook to disk. Maintains kernel state between calls so variables
        and imports persist across executions.

        Args:
            code (str): The code or markdown content to run.
            language (Literal["python", "markdown"], optional): The language type. Defaults to "python".

        Returns:
            Observation: An object containing:
                - is_success: Success status (True if execution succeeded)
                - message: The execution output text or markdown content
                - base64_images: List of base64 encoded images generated during execution

        Raises:
            ValueError: If language is not "python" or "markdown".
        """
        self._display(code, language)
        output: str = ""
        success: bool = False
        images: List[str] = []

        try:
            if language == "python":
                # add code to the notebook
                self.add_code_cell(code=code)

                # run code with our persistent kernel
                cell_index = len(self.nb.cells) - 1
                success, output, images = await self.run_cell(
                    self.nb.cells[-1], cell_index
                )

                # Handle special cases
                if "!pip" in code:
                    success = False
                    output = output[-INSTALL_KEEPLEN:]
                elif "git clone" in code:
                    output = (
                        output[:INSTALL_KEEPLEN] + "..." + output[-INSTALL_KEEPLEN:]
                    )

            elif language == "markdown":
                # add markdown content to markdown cell in a notebook.
                self.add_markdown_cell(code)
                # return success for markdown cell.
                output, success = code, True
            else:
                raise ValueError(
                    f"Only support for language: python, markdown, but got {language}, "
                )
        except KeyboardInterrupt:
            logger.warning("Operation interrupted by user")
            output = "Operation interrupted by user"
            success = False
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            output = f"Error: {str(e)}"
            success = False
        finally:
            # Save the notebook even if there's an error
            try:
                file_path = self.workspace / "code.ipynb"
                nbformat.write(self.nb, file_path)
            except Exception as e:
                logger.error(f"Error saving notebook: {str(e)}")

        # Return the result using our structured model
        return Observation(
            is_success=success,
            message=output,
            base64_images=[f"data:image/png;base64,{image}" for image in images],
        )

    async def __aenter__(self):
        """Async context manager entry point."""
        await self.init_kernel()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point that ensures proper cleanup."""
        await self.terminate()
        return False  # Don't suppress exceptions


def filter_log_lines(input_str: str) -> str:
    """
    Remove log and warning lines from the output string.

    Args:
        input_str (str): The input string potentially containing log and warning lines.

    Returns:
        str: The cleaned string with log and warning lines removed.
    """
    delete_lines = ["[warning]", "warning:", "[cv]", "[info]"]
    result = "\n".join(
        [
            line
            for line in input_str.split("\n")
            if not any(dl in line.lower() for dl in delete_lines)
        ]
    ).strip()
    return result


def strip_ansi_codes(input_str: str) -> str:
    """
    Remove ANSI escape sequences and color codes from text output.

    Args:
        input_str (str): String containing ANSI escape and color codes.

    Returns:
        str: Clean string with escape and color codes removed.
    """
    # Use regular expressions to get rid of escape characters and color codes in jupyter notebook output.
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    result = pattern.sub("", input_str)
    return result


def render_markdown(content: str) -> None:
    """
    Display markdown content with proper formatting.

    Parses markdown content, separating text and code blocks, and displays them
    with appropriate styling using Rich library.

    Args:
        content (str): The markdown content to display.
    """
    # Use regular expressions to match blocks of code one by one.
    matches = re.finditer(r"```(.+?)```", content, re.DOTALL)
    start_index = 0
    content_panels = []
    # Set the text background color and text color.
    style = "black on white"
    # Print the matching text and code one by one.
    for match in matches:
        text_content = content[start_index : match.start()].strip()
        code_content = match.group(0).strip()[3:-3]  # Remove triple backticks

        if text_content:
            content_panels.append(
                Panel(Markdown(text_content), style=style, box=MINIMAL)
            )

        if code_content:
            content_panels.append(
                Panel(Markdown(f"```{code_content}"), style=style, box=MINIMAL)
            )
        start_index = match.end()

    # Print remaining text (if any).
    remaining_text = content[start_index:].strip()
    if remaining_text:
        content_panels.append(Panel(Markdown(remaining_text), style=style, box=MINIMAL))

    # Display all panels in Live mode.
    with Live(
        auto_refresh=False, console=Console(), vertical_overflow="visible"
    ) as live:
        live.update(Group(*content_panels))
        live.refresh()
