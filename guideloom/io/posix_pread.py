from collections.abc import Callable
from typing import final
import os
import asyncio
import pathlib
import contextlib

from ..types import IOAsyncReaderAbstract, IOReaderAbstract, IOSyncReaderAbstract

_IS_POSIX = os.name == "posix"

_HAS_FADVICE = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM")


@final
class PosixPreadIO(IOReaderAbstract):
    """Read shard bytes with `os.preadv`.

    Available on POSIX platforms that provide `os.preadv`.
    """

    def __init__(self, *, advice: bool = False) -> None:
        """Create a preadv backend.

        Args:
            advice (bool): When True, call `posix_fadvise` with `POSIX_FADV_RANDOM` after each file
                is opened. Defaults to False.

        Raises:
            OSError: If `os.preadv` is unavailable, or if `advice` is True and `posix_fadvise` is
                unavailable.
        """
        if not _IS_POSIX or not hasattr(os, "preadv"):
            raise OSError("PosixPreadIO requires os.preadv on a POSIX platform")
        if advice and not _HAS_FADVICE:
            raise OSError("posix_fadvise is not supported on this platform use a different I/O interface")

        super().__init__()
        self.advice = advice

    @staticmethod
    def is_supported(*, advice: bool = False) -> bool:
        """Return whether this platform can construct `PosixPreadIO`.

        Args:
            advice (bool): Also require `posix_fadvise`. Defaults to False.

        Returns:
            bool: True when `os.preadv` exists, and `posix_fadvise` exists if `advice` is True.
        """
        if not _IS_POSIX or not hasattr(os, "preadv"):
            return False
        return not (advice and not _HAS_FADVICE)

    def sync_reader(self) -> Callable[[pathlib.PurePath], "_PosixPreadIOSync"]:
        """Return a constructor for a synchronous reader of one file.

        The constructor receives the `advice` setting from this backend.

        Returns:
            Callable[[pathlib.PurePath], IOSyncReaderAbstract]: Called with the file path.
        """
        advice = self.advice

        def make(path: pathlib.PurePath) -> _PosixPreadIOSync:
            return _PosixPreadIOSync(path, advice=advice)

        return make

    def async_reader(self) -> Callable[[pathlib.PurePath], "_PosixPreadIOAsync"]:
        """Return a constructor for an asynchronous reader of one file.

        The constructor receives the `advice` setting from this backend.

        Returns:
            Callable[[pathlib.PurePath], IOAsyncReaderAbstract]: Called with the file path.
        """
        advice = self.advice

        def make(path: pathlib.PurePath) -> _PosixPreadIOAsync:
            return _PosixPreadIOAsync(path, advice=advice)

        return make


class _PosixPreadIOShare:
    def __init__(self, path: pathlib.PurePath, *, advice: bool = False) -> None:
        self._path = path
        self._advice = advice
        self._fd: int | None = None

    def _open(self) -> None:
        self._fd = os.open(self._path, os.O_RDONLY)
        if self._advice:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)  # type: ignore

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise OSError("not open")
        return self._fd

    def _close(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)


@final
class _PosixPreadIOSync(_PosixPreadIOShare, IOSyncReaderAbstract):
    def filesize(self) -> int:
        return os.stat(self.fd).st_size

    def read(self, offset: int, size: int) -> bytearray:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")
        content = bytearray(size)
        bytesread = os.preadv(self.fd, (content,), offset)
        if bytesread < size:
            raise IndexError(f"read content ({bytesread}B) was shorter than requested size ({size}B)")
        return content

    def open(self) -> None:
        super()._open()

    def close(self) -> None:
        super()._close()


@final
class _PosixPreadIOAsync(_PosixPreadIOShare, IOAsyncReaderAbstract):
    async def filesize(self) -> int:
        return (await asyncio.to_thread(os.stat, self.fd)).st_size

    async def read(self, offset: int, size: int) -> bytearray:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")
        content = bytearray(size)
        bytesread = await asyncio.to_thread(os.preadv, self.fd, (content,), offset)
        if bytesread < size:
            raise IndexError(f"read content ({bytesread}B) was shorter than requested size ({size}B)")
        return content

    async def open(self) -> None:
        super()._open()

    async def close(self) -> None:
        super()._close()

    def close_sync(self) -> None:
        super()._close()
