from abc import ABC, abstractmethod
from pathlib import Path

class FileStore(ABC):
    @abstractmethod
    def write(self, path: str | Path, contents: str | bytes) -> None:
        pass

    @abstractmethod
    def read(self, path: str | Path, binary: bool = False) -> str | bytes:
        pass

    @abstractmethod
    def list(self, path: str | Path) -> list[Path]:
        pass

    @abstractmethod
    def delete(self, path: str | Path) -> None:
        pass