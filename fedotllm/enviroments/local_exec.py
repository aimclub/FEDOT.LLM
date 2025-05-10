import contextlib
import faulthandler
import io
import multiprocessing
import os
import pickle
import platform
import shutil
import signal
import sys
import tempfile
import traceback
from .types import ProgramStatus, ExecutionResult
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional


class TimeoutException(Exception):
    pass


def filter_picklable(d):
    return {k: v for k, v in d.items() if is_picklable(v)}


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


class Tee:
    """
    A helper class that duplicates stream writes to multiple streams or sends data through connections.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if hasattr(stream, "write"):
                try:
                    stream.write(data)
                except AttributeError:
                    pass  # Ignore if the stream doesn't support write
            elif isinstance(stream, Connection):
                try:
                    stream.send(data)
                except (EOFError, BrokenPipeError):
                    pass  # Handle closed connections gracefully

    def flush(self):
        for stream in self.streams:
            if hasattr(stream, "flush"):
                try:
                    stream.flush()
                except AttributeError:
                    pass  # Ignore streams that don't support flush

    def getvalue(self):
        """
        Aggregates the output from all streams that support `getvalue`.
        Note: Connection objects do not support `getvalue`.
        """
        values = []
        for stream in self.streams:
            if hasattr(stream, "getvalue"):
                try:
                    values.append(stream.getvalue())
                except AttributeError:
                    pass  # Ignore if getvalue raises an exception
        return "".join(values)


def execute_code(
    code,
    timeout,
    sandbox=True,
    in_glob=None,
    argv=None,
    output_dir: Optional[Path] = None,
    vaults: Optional[list[Path]] = None,
    show_progress=True,
):
    """
    Executes the provided code with optional sandboxing, vaults, and progress display.

    Args:
        code (str): The code to execute.
        timeout (int): Maximum execution time in seconds.
        sandbox (bool): Whether to execute in a sandboxed environment.
        in_glob (dict): Global variables to use in the code. Defaults to {}.
        argv: Command-line arguments to pass to the code. Defaults to None.
        output_dir: Directory to store output files. Defaults to None.
        vaults (list[Path], optional): List of paths to vaults to include. Defaults to None.
        show_progress (bool, optional): Whether to display execution progress. Defaults to True.

    Returns:
        ExecutionResult: The result of the execution.
    """
    if argv is None:
        argv = []
    if in_glob is None:
        in_glob = {}

    if sandbox:
        manager = multiprocessing.Manager()
        result = manager.list()
        pipes = (
            (multiprocessing.Pipe(), multiprocessing.Pipe()) if show_progress else None
        )

        process_args = (
            code,
            timeout,
            result,
            in_glob,
            argv,
            output_dir,
            vaults,
            (pipes[0][1], pipes[1][1]) if pipes else None,
        )
        process = multiprocessing.Process(target=unsafe_execute, args=process_args)
        process.start()

        if show_progress and pipes:
            parent_stdout, child_stdout = pipes[0]
            parent_stderr, child_stderr = pipes[1]

            # Close child ends in the parent process
            child_stdout.close()
            child_stderr.close()

            try:
                while True:
                    if parent_stdout.poll(0.1):
                        msg = parent_stdout.recv()
                        if msg is None:
                            break
                        print(msg, end="")

                    if parent_stderr.poll(0.1):
                        msg = parent_stderr.recv()
                        if msg is None:
                            break
                        print(msg, end="", file=sys.stderr)

                    if not process.is_alive():
                        break
            except EOFError:
                pass

        # Wait for the process to finish or timeout
        print(
            f"Waiting for process cleanup with timeout {timeout if timeout < 5 * 60 else 5 * 60}..."
        )
        process.join(timeout=timeout if timeout < 5 * 60 else 5 * 60)
        if process.is_alive():
            process.kill()

        result = result._getvalue()
    else:
        result = []
        unsafe_execute(code, timeout, result, in_glob)

    return result[0] if result else None


SCRIPT_NAME = "solution.py"


def unsafe_execute(
    code: str,
    timeout: int,
    result: list,
    in_glob=None,
    argv=None,
    output_dir: Optional[Path] = None,
    vaults: Optional[list[Path]] = None,
    pipes: Optional[tuple[Connection, Connection]] = None,
):
    if argv is None:
        argv = []
    if in_glob is None:
        in_glob = {}

    with create_tempdir(vaults, output_dir):
        # These system calls are needed when cleaning up tempdir.
        import os

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        with set_argv(argv):
            # Disable functionalities that can make destructive changes to the test.
            # reliability_guard()
            exec_globals = {
                "__name__": "__main__",
                "__file__": SCRIPT_NAME,
            } | in_glob
            exec_result = ExecutionResult()
            tracing = io.StringIO()
            tracing.seek(0)
            with open(SCRIPT_NAME, "w") as f:
                f.write(code)
            try:
                with swallow_io(pipes=pipes) as (stdout_stream, stderr_stream):
                    with time_limit(timeout):
                        exec(compile(code, SCRIPT_NAME, "exec"), exec_globals)
                exec_result.program_status = ProgramStatus.kSuccess
            except TimeoutException:
                exec_result.program_status = ProgramStatus.kTimeout
            except BaseException:
                clean_exception = True
                filtered_trace = traceback.format_exc()
                if clean_exception:
                    filtered_trace = filtered_trace.splitlines()
                    filtered_trace = filtered_trace[min(len(filtered_trace) - 2, 3) :]
                    filtered_trace = "\n".join(filtered_trace)
                exec_result.sandbox_result = filtered_trace
                exec_result.program_status = ProgramStatus.kFailed
            finally:
                captured_output = stdout_stream.getvalue().strip()
                exec_result.stderr = stderr_stream.getvalue()
                exec_result.stdout = captured_output
                exec_result.trace = tracing.getvalue()
                exec_result.global_vars = filter_picklable(exec_globals)
        result.append(exec_result)
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def set_argv(argv: list):
    """
    Sets the sys.argv to the provided list and restores the original sys.argv after the context manager exits.
    """
    original_argv = sys.argv.copy()
    sys.argv = [SCRIPT_NAME] + argv
    try:
        yield
    finally:
        sys.argv = original_argv


@contextlib.contextmanager
def time_limit(seconds):
    """
    Sets a timer to raise a TimeoutException after the specified number of seconds.
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io(
    input_stream=None,
    binary=False,
    pipes: Optional[tuple[Connection, Connection]] = None,
):
    """
    Swallows the standard input, output, and error streams and optionally redirects them through pipes.
    """
    # Save the original stdin, stdout, and stderr.
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Replace standard streams with in-memory buffers
    if input_stream is not None:
        sys.stdin = input_stream if binary else io.TextIOWrapper(input_stream)
    else:
        sys.stdin = io.StringIO() if not binary else io.BytesIO()

    sys.stdout = io.BytesIO() if binary else io.StringIO()
    sys.stderr = io.BytesIO() if binary else io.StringIO()

    if pipes:
        sys.stdout = Tee(sys.stdout, pipes[0])
        sys.stderr = Tee(sys.stderr, pipes[1])

    try:
        yield sys.stdout, sys.stderr
    finally:
        # Restore the original stdin, stdout, and stderr.
        sys.stdin = original_stdin
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Close the pipes by sending None
        if pipes:
            try:
                pipes[0].send(None)
                pipes[1].send(None)
            except Exception:
                pass


@contextlib.contextmanager
def create_tempdir(
    vaults: Optional[list[Path]] = None, output_dir: Optional[Path] = None
):
    """
    Creates a temporary directory and optionally links vaults to it.
    """
    if output_dir:
        output_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as dirname:
        temp_dir = Path(dirname)
        # Copy datasets to temp directory
        if vaults:
            for vault in vaults:
                vault_name = vault.name
                temp_vault_path = temp_dir / vault_name
                try:
                    # Create a symbolic link pointing to the vault
                    temp_vault_path.symlink_to(
                        vault.resolve(), target_is_directory=True
                    )
                    # Change permissions to read-only if supported
                    os.chmod(temp_vault_path, 0o555)
                    print(f"Linked vault {vault} to {temp_vault_path} as read-only.")
                except Exception as e:
                    print(f"Failed to link vault {vault}: {e}")
                    raise
        with chdir(dirname):
            yield dirname
        if output_dir:
            # Get the relative paths and copy maintaining structure
            for root, dirs, files in os.walk(dirname):
                # Calculate relative path from dirname
                rel_path = Path(root).relative_to(dirname)

                if rel_path == Path("."):
                    # Handle directories
                    for dir in dirs:
                        if vaults and any(vault.name in dir for vault in vaults):
                            continue
                        src_dir = Path(root) / dir
                        dst_dir = output_dir / rel_path / dir
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                        print(f"Copied {src_dir} to {dst_dir}")

                    # Handle files
                    for file in files:
                        src_file = Path(root) / file
                        dst_file = output_dir / file
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src_file, dst_file)
                        print(f"Copied {src_file} to {dst_file}")


@contextlib.contextmanager
def chdir(root):
    """
    Changes the current working directory to the provided root and restores the original directory after the context manager exits.
    """
    original_dir = os.getcwd()
    if root == ".":
        yield
        return
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(original_dir)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-generated
    code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    # os.putenv=None used by Numpy
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    # os.getcwd=None used by PyTorch
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    # sys.modules["joblib"]=None used by scikit-learn
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


if __name__ == "__main__":
    pass
