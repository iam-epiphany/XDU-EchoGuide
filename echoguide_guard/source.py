"""Read-only source adapters used by the EchoGuide Guard static scanner.

The scanner accepts either a directory or a ZIP archive.  ZIP members are read
in-place: the archive is never extracted, which keeps a scan from writing
untrusted paths to disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Union
import os
import zipfile


DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_FILES = 10_000


class SourceLimitError(ValueError):
    """Raised when a source exceeds a scanner safety limit."""


@dataclass(frozen=True)
class SourceFile:
    """A source-relative file and its immutable content."""

    path: str
    content: bytes

    @property
    def size(self) -> int:
        return len(self.content)

    def read_text(self) -> str:
        """Decode common UTF text while tolerating a malformed byte sequence."""

        if self.content.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return self.content.decode("utf-16")
            except UnicodeDecodeError:
                pass
        return self.content.decode("utf-8-sig", errors="replace")


class ScanSource(ABC):
    """Common interface for directory and archive-backed scan inputs."""

    kind: str

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_files = max_files

    @abstractmethod
    def iter_files(self) -> Iterator[SourceFile]:
        """Yield safe, source-relative files in deterministic order."""

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.path.name,
            "path": str(self.path),
        }

    def _check_running_limits(self, file_count: int, total_bytes: int) -> None:
        if file_count > self.max_files:
            raise SourceLimitError(
                f"source contains more than {self.max_files} readable files"
            )
        if total_bytes > self.max_total_bytes:
            raise SourceLimitError(
                f"source expands beyond {self.max_total_bytes} bytes"
            )


class DirectorySource(ScanSource):
    kind = "directory"

    # Dependency and VCS trees add noise and can dwarf the application source.
    _SKIPPED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }

    def __init__(self, path: Union[str, os.PathLike[str]], **limits: int) -> None:
        super().__init__(path, **limits)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        if not self.path.is_dir():
            raise ValueError(f"scan source is not a directory: {self.path}")

    def iter_files(self) -> Iterator[SourceFile]:
        paths = []
        for current, dirnames, filenames in os.walk(self.path, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in self._SKIPPED_DIRS
                and not Path(current, name).is_symlink()
            )
            for filename in sorted(filenames):
                candidate = Path(current, filename)
                if not candidate.is_symlink() and candidate.is_file():
                    paths.append(candidate)

        total_bytes = 0
        file_count = 0
        for candidate in sorted(paths, key=lambda item: item.as_posix().lower()):
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size > self.max_file_bytes:
                continue
            try:
                content = candidate.read_bytes()
            except OSError:
                continue
            file_count += 1
            total_bytes += len(content)
            self._check_running_limits(file_count, total_bytes)
            relative = candidate.relative_to(self.path).as_posix()
            yield SourceFile(relative, content)


class ZipSource(ScanSource):
    kind = "zip"

    def __init__(self, path: Union[str, os.PathLike[str]], **limits: int) -> None:
        super().__init__(path, **limits)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        if not self.path.is_file() or not zipfile.is_zipfile(self.path):
            raise ValueError(f"scan source is not a valid ZIP archive: {self.path}")

    @staticmethod
    def _safe_member_name(name: str) -> str | None:
        normalized = name.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or not pure.parts:
            return None
        if any(part in {"", ".", ".."} for part in pure.parts):
            return None
        if pure.parts[0].endswith(":") or "__MACOSX" in pure.parts:
            return None
        return pure.as_posix()

    def iter_files(self) -> Iterator[SourceFile]:
        total_bytes = 0
        file_count = 0
        with zipfile.ZipFile(self.path, "r") as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.filename.lower())
            for info in infos:
                if info.is_dir():
                    continue
                relative = self._safe_member_name(info.filename)
                if relative is None:
                    continue
                parts = PurePosixPath(relative).parts
                if any(part in DirectorySource._SKIPPED_DIRS for part in parts):
                    continue
                if info.file_size > self.max_file_bytes:
                    continue
                # A high compression ratio is a useful cheap ZIP-bomb guard.
                if info.file_size > 1024 * 1024 and info.compress_size:
                    if info.file_size / info.compress_size > 200:
                        continue
                try:
                    content = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    continue
                file_count += 1
                total_bytes += len(content)
                self._check_running_limits(file_count, total_bytes)
                yield SourceFile(relative, content)


def open_source(
    source: Union[str, os.PathLike[str], ScanSource],
    **limits: int,
) -> ScanSource:
    """Return a source adapter for a directory or ZIP archive."""

    if isinstance(source, ScanSource):
        if limits:
            raise ValueError("limits cannot be overridden for an existing ScanSource")
        return source
    path = Path(source).expanduser()
    if path.is_dir():
        return DirectorySource(path, **limits)
    if path.is_file() and zipfile.is_zipfile(path):
        return ZipSource(path, **limits)
    if not path.exists():
        raise FileNotFoundError(str(path))
    raise ValueError(f"scan source must be a directory or ZIP archive: {path}")


# A descriptive alias for callers that prefer a factory-style name.
source_from_path = open_source


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DirectorySource",
    "ScanSource",
    "SourceFile",
    "SourceLimitError",
    "ZipSource",
    "open_source",
    "source_from_path",
]
