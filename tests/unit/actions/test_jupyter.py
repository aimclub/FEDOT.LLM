import pytest
import os
import asyncio
import json
import base64
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from fedotllm.environments.jupyter import (
    JupyterExecutor, 
    JupyterExecutionResult,
    strip_ansi_codes,
    filter_log_lines,
    render_markdown
)


class TestJupyterExecutor:
    """Test suite for the JupyterExecutor class"""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Fixture that provides a temporary workspace directory."""
        return tmp_path

    @pytest.fixture
    def executor(self, temp_workspace):
        """Returns a function that creates a JupyterExecutor instance."""
        return lambda: JupyterExecutor(temp_workspace)

    async def execute_with_cleanup(self, executor_factory, test_func):
        """Execute a test function with an executor and ensure cleanup."""
        executor = executor_factory()
        try:
            await test_func(executor)
        finally:
            await executor.terminate()

    @pytest.mark.asyncio
    async def test_variable_persistence(self, executor):
        """Test that variables persist between executions."""
        async def run_test(executor):
            # Assign a variable
            result1 = await executor.run('x = 42')
            assert result1.is_success, f"Failed to assign variable: {result1.output}"
            
            # Verify the variable persists
            result2 = await executor.run('x')
            assert result2.is_success, f"Failed to access variable: {result2.output}"
            assert '42' in result2.output, f"Expected '42' in output, got: {result2.output}"
        
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_function_definition_and_use(self, executor):
        """Test that function definitions persist and can be used."""
        async def run_test(executor):
            # Define a function
            result1 = await executor.run('def square(x): return x * x')
            assert result1.is_success, f"Failed to define function: {result1.output}"
            
            # Assign a variable to use with the function
            result2 = await executor.run('num = 7')
            assert result2.is_success, f"Failed to assign variable: {result2.output}"
            
            # Call the function with the variable
            result3 = await executor.run('square(num)')
            assert result3.is_success, f"Failed to call function: {result3.output}"
            assert '49' in result3.output, f"Expected '49' in output, got: {result3.output}"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_imports_persistence(self, executor):
        """Test that imported modules persist between executions."""
        async def run_test(executor):
            # Import a module
            result1 = await executor.run('import math')
            assert result1.is_success, f"Failed to import module: {result1.output}"
            
            # Use the imported module
            result2 = await executor.run('math.sqrt(25)')
            assert result2.is_success, f"Failed to use imported module: {result2.output}"
            assert '5.0' in result2.output, f"Expected '5.0' in output, got: {result2.output}"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_error_handling(self, executor):
        """Test handling of code that generates an error."""
        async def run_test(executor):
            # Code with syntax error
            result1 = await executor.run('print(')
            assert not result1.is_success, "Expected failure for syntax error"
            assert "SyntaxError" in result1.output, f"Expected 'SyntaxError' in output, got: {result1.output}"
            
            # Code with runtime error
            result2 = await executor.run('1/0')
            assert not result2.is_success, "Expected failure for division by zero"
            assert "ZeroDivisionError" in result2.output, f"Expected 'ZeroDivisionError' in output, got: {result2.output}"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_plot_generation(self, executor):
        """Test that plots generate images correctly."""
        async def run_test(executor):
            # Create a simple matplotlib plot
            plot_code = '''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.grid(True)
plt.show()
'''
            result = await executor.run(plot_code)
            assert result.is_success, f"Failed to create plot: {result.output}"
            assert len(result.images) > 0, "No images were captured"
            
            # Verify the image is a valid PNG
            image_bytes = base64.b64decode(result.images[0])
            assert image_bytes.startswith(b'\x89PNG'), "Not a valid PNG image"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_markdown_rendering(self, executor):
        """Test rendering of markdown cells."""
        async def run_test(executor):
            markdown = """# Test Heading
This is a test of *markdown* rendering with **bold** text.

- List item 1
- List item 2

```python
def hello():
    print('Hello, world!')
```
"""
            result = await executor.run(markdown, language="markdown")
            assert result.is_success, f"Failed to render markdown: {result.output}"
            assert result.output == markdown, "Markdown output should match input"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_kernel_reset(self, executor):
        """Test that kernel reset clears all variables."""
        async def run_test(executor):
            # Define a variable
            result1 = await executor.run('test_var = "This should be cleared"')
            assert result1.is_success, f"Failed to define variable: {result1.output}"
            
            # Verify the variable exists
            result2 = await executor.run('test_var')
            assert result2.is_success, f"Failed to access variable: {result2.output}"
            assert "This should be cleared" in result2.output
            
            # Reset the kernel
            await executor.reset()
            
            # Verify the variable no longer exists
            result3 = await executor.run('test_var')
            assert not result3.is_success, "Variable should not persist after reset"
            assert "NameError" in result3.output, "Expected NameError after kernel reset"
            
        await self.execute_with_cleanup(executor, run_test)

    @pytest.mark.asyncio
    async def test_notebook_file_creation(self, executor):
        """Test that a notebook file is created after execution."""
        async def run_test(executor):
            await executor.run('print("Testing notebook creation")')
            
            # Check that the notebook file exists
            notebook_path = executor.workspace / "code.ipynb"
            assert notebook_path.exists(), "Notebook file was not created"
            
            # Verify it has the expected content
            import nbformat
            nb = nbformat.read(notebook_path, as_version=4)
            assert len(nb.cells) > 0, "Notebook has no cells"
            assert 'print("Testing notebook creation")' in nb.cells[-1].source
            
        await self.execute_with_cleanup(executor, run_test)


class TestJupyterExecutionResult:
    """Test suite for the JupyterExecutionResult class"""
    
    @pytest.fixture
    def successful_result(self):
        """Fixture that provides a successful execution result."""
        return JupyterExecutionResult(
            is_success=True,
            output="Test output",
            images=[]
        )
    
    @pytest.fixture
    def result_with_image(self):
        """Fixture that provides an execution result with an image."""
        # Create a simple image (1x1 white pixel)
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        
        # Save to bytes and encode
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return JupyterExecutionResult(
            is_success=True,
            output="Test output with image",
            images=[image_b64]
        )
    
    @pytest.fixture
    def error_result(self):
        """Fixture that provides an error execution result."""
        return JupyterExecutionResult(
            is_success=False,
            output="Test error output: ZeroDivisionError: division by zero",
            images=[]
        )
    
    def test_to_dict(self, successful_result, error_result, result_with_image):
        """Test the to_dict method"""
        # Test successful result
        d1 = successful_result.to_dict()
        assert d1["is_success"] is True
        assert d1["output"] == "Test output"
        assert len(d1["images"]) == 0
        
        # Test error result
        d2 = error_result.to_dict()
        assert d2["is_success"] is False
        assert "ZeroDivisionError" in d2["output"]
        
        # Test result with image
        d3 = result_with_image.to_dict()
        assert d3["is_success"] is True
        assert len(d3["images"]) == 1
    
    def test_to_markdown(self, successful_result, error_result, result_with_image):
        """Test the to_markdown method"""
        # Test successful result
        md1 = successful_result.to_markdown()
        assert "✅ Success" in md1
        assert "Test output" in md1
        
        # Test error result
        md2 = error_result.to_markdown()
        assert "❌ Error" in md2
        assert "ZeroDivisionError" in md2
        
        # Test result with image
        md3 = result_with_image.to_markdown()
        assert "✅ Success" in md3
        assert "**Images**: 1 image(s)" in md3
        assert "![Execution Result Image 1](data:image/png;base64," in md3
    
    def test_for_llm(self, successful_result, result_with_image):
        """Test the for_llm method"""
        # Test successful result
        llm1 = successful_result.for_llm()
        assert llm1["execution_successful"] is True
        assert llm1["text_output"] == "Test output"
        assert llm1["has_images"] is False
        assert llm1["image_count"] == 0
        
        # Test result with image
        llm2 = result_with_image.for_llm()
        assert llm2["execution_successful"] is True
        assert llm2["has_images"] is True
        assert llm2["image_count"] == 1
        assert "image_references" in llm2
    
    def test_get_images_as_data_uris(self, result_with_image):
        """Test the get_images_as_data_uris method"""
        uris = result_with_image.get_images_as_data_uris()
        assert len(uris) == 1
        assert uris[0].startswith("data:image/png;base64,")


class TestUtilityFunctions:
    """Test suite for utility functions in the jupyter module"""
    
    def test_strip_ansi_codes(self):
        """Test stripping ANSI escape codes from text"""
        test_input = "\x1b[31mError:\x1b[0m Something went wrong"
        result = strip_ansi_codes(test_input)
        assert result == "Error: Something went wrong"
    
    def test_filter_log_lines(self):
        """Test filtering log lines from output"""
        test_input = "Line 1\n[warning] This is a warning\nLine 2\n[info] Information\nLine 3"
        result = filter_log_lines(test_input)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "[warning]" not in result
        assert "[info]" not in result 