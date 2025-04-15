from pathlib import Path
from fedotllm.storage.base import FileStore


class LocalFileStore(FileStore):
    root: Path

    def __init__(self, root: str | Path):
        if isinstance(root, str):
            self.root = Path(root)
        else:
            self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_full_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            path = Path(path)
        full_path = self.root / path
        # Make sure we don't escape the root directory
        if not full_path.is_relative_to(self.root):
            raise ValueError(f"Path {path} attempts to escape the root directory")
        return full_path

    def write(self, path: str | Path, contents: str | bytes) -> None:
        full_path = self.get_full_path(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if isinstance(contents, str) else "wb"
            with open(full_path, mode) as f:
                f.write(contents)
        except PermissionError:
            raise PermissionError(f"No permission to write to {full_path}")
        except IOError as e:
            raise IOError(f"IO error writing to {full_path}: {e}")

    def read(self, path: str | Path, binary: bool = False) -> str | bytes:
        full_path = self.get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        mode = "rb" if binary else "r"
        with open(full_path, mode) as f:
            return f.read()

    def list(self, path: str | Path) -> list[Path]:
        full_path = self.get_full_path(path)
        paths = list(full_path.glob("**/*"))
        return [p.relative_to(self.root) for p in paths]

    def delete(self, path: str | Path) -> None:
        full_path = self.get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            for child in full_path.iterdir():
                self.delete(child)
            full_path.rmdir()
        else:
            raise FileNotFoundError(f"Path {path} is not a file or directory")
