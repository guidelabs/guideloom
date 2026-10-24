from collections.abc import Callable
from typing import final
import os
import pathlib

import aiofiles
import aiofiles.os

from ..types import IOAsyncReaderAbstract, IOReaderAbstract, IOSyncReaderAbstract


@final
class SeekIO(IOReaderAbstract[pathlib.Path]):
    """Read shard bytes by seeking within the file for each read.

    This backend is supported on every platform. Storage uses it when `read_interface` is omitted.
    """

    @staticmethod
    def is_supported() -> bool:
        """Return whether this backend can be constructed.

        Returns:
            bool: Always True.
        """
        return True

    def sync_reader(self) -> Callable[[pathlib.Path], "_SeekIOSync"]:
        """Return a constructor for a synchronous reader of one file.

        Returns:
            Callable[[pathlib.Path], IOSyncReaderAbstract]: Called with the file path.
        """
        return _SeekIOSync

    def async_reader(self) -> Callable[[pathlib.Path], "_SeekIOAsync"]:
        """Return a constructor for an asynchronous reader of one file.

        Returns:
            Callable[[pathlib.Path], IOAsyncReaderAbstract]: Called with the file path.
        """
        return _SeekIOAsync


class _SeekIOShare:
    def __init__(self, path: pathlib.Path) -> None:
        self._path = path


@final
class _SeekIOSync(_SeekIOShare, IOSyncReaderAbstract):
    def filesize(self) -> int:
        return os.stat(self._path).st_size

    def read(self, offset: int, size: int) -> bytearray | memoryview:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")

        content = bytearray(size)
        with open(self._path, "rb") as f:
            f.seek(offset)
            bytesread = f.readinto(content)
        if bytesread < size:
            raise IndexError(f"read content ({bytesread}B) was shorter than requested size ({size}B)")
        return content

    def open(self) -> None:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"the file {self._path} does not exist")

    def close(self) -> None:
        pass


@final
class _SeekIOAsync(_SeekIOShare, IOAsyncReaderAbstract):
    async def filesize(self) -> int:
        return (await aiofiles.os.stat(self._path)).st_size

    async def read(self, offset: int, size: int) -> bytearray | memoryview:
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")

        content = bytearray(size)
        async with aiofiles.open(self._path, "rb") as f:
            await f.seek(offset)
            bytesread = await f.readinto(content)
        assert bytesread is not None
        if bytesread < size:
            raise IndexError(f"read content ({bytesread}B) was shorter than requested size ({size}B)")
        return content

    async def open(self) -> None:
        if not await aiofiles.os.path.exists(self._path):
            raise FileNotFoundError(f"the file {self._path} does not exist")

    async def close(self) -> None:
        pass

    def close_sync(self) -> None:
        pass
