from collections.abc import Callable
from operator import getitem
from typing import final
import io
import os
import mmap
import asyncio
import pathlib
import contextlib

from ..types import IOAsyncReaderAbstract, IOReaderAbstract, IOSyncReaderAbstract


@final
class MmapIO(IOReaderAbstract[pathlib.Path]):
    """Read shard bytes from a read-only memory map.

    This backend is supported on every platform. Each read copies the requested range out of the map.
    """

    @staticmethod
    def is_supported() -> bool:
        """Return whether this backend can be constructed.

        Returns:
            bool: Always True.
        """
        return True

    def sync_reader(self) -> Callable[[pathlib.Path], "_MmapIOSync"]:
        """Return a constructor for a synchronous reader of one file.

        Returns:
            Callable[[pathlib.Path], IOSyncReaderAbstract]: Called with the file path.
        """
        return _MmapIOSync

    def async_reader(self) -> Callable[[pathlib.Path], "_MmapIOAsync"]:
        """Return a constructor for an asynchronous reader of one file.

        Returns:
            Callable[[pathlib.Path], IOAsyncReaderAbstract]: Called with the file path.
        """
        return _MmapIOAsync


class _MmapIOShare:
    def __init__(self, path: pathlib.Path) -> None:
        self._fd: io.FileIO | None = None
        self._mmap: mmap.mmap | bytes | None = None
        self._size: int | None = None
        self._path = path

    def _open(self) -> None:
        self._fd = open(self._path, "rb", buffering=0)  # noqa: SIM115
        self._size = os.stat(self.fd.fileno()).st_size
        if self._size == 0:
            self._mmap = b""
        else:
            self._mmap = mmap.mmap(self._fd.fileno(), self._size, access=mmap.ACCESS_READ)

    def _close(self) -> None:
        with contextlib.suppress(OSError, AttributeError):
            if isinstance(self.mmap, mmap.mmap):
                self.mmap.close()
        with contextlib.suppress(OSError, AttributeError):
            self.fd.close()

    @property
    def fd(self) -> io.FileIO:
        if self._fd is None:
            raise OSError("not open")
        return self._fd

    @property
    def mmap(self) -> mmap.mmap | bytes:
        if self._mmap is None:
            raise OSError("not open")
        return self._mmap

    @property
    def size(self) -> int:
        if self._size is None:
            raise OSError("not open")
        return self._size


@final
class _MmapIOSync(_MmapIOShare, IOSyncReaderAbstract):
    def filesize(self) -> int:
        return self.size

    def read(self, offset: int, size: int) -> bytearray:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")
        content = bytearray(self.mmap[offset : offset + size])
        if len(content) < size:
            raise IndexError(f"read content ({len(content)}B) was shorter than requested size ({size}B)")
        return content

    def open(self) -> None:
        super()._open()

    def close(self) -> None:
        super()._close()


@final
class _MmapIOAsync(_MmapIOShare, IOAsyncReaderAbstract):
    async def filesize(self) -> int:
        return self.size

    async def read(self, offset: int, size: int) -> bytearray:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")
        sl = slice(offset, offset + size)
        content = bytearray(await asyncio.to_thread(getitem, self._mmap, sl))  # pyright: ignore[reportArgumentType]
        if len(content) < size:
            raise IndexError(f"read content ({len(content)}B) was shorter than requested size ({size}B)")
        return content

    async def open(self) -> None:
        super()._open()

    async def close(self) -> None:
        super()._close()

    def close_sync(self) -> None:
        super()._close()
