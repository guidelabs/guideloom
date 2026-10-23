from collections import defaultdict
from typing import BinaryIO, Final, TypeIs, final, overload
import os
import asyncio
import pathlib
import threading

import numpy as np

from ..io import SeekIO
from ..types import (
    AsyncContextManager,
    IOAsyncReaderAbstract,
    IOReaderAbstract,
    IOSyncReaderAbstract,
    SyncContextManager,
    Vector,
)
from ._shard_count import num_shards_async, num_shards_sync


def _shard_to_filename(shard: int) -> str:
    return f"{shard:07d}.in_band.bin"


@final
class FileSystemStream[PathT: pathlib.PurePath = pathlib.Path]:
    """Read and write a stream of NumPy arrays in a directory of shard files.

    `path` and `read_interface` are immutable and share `PathT`. Built-in backends require a
    `pathlib.Path`. A custom backend can require `pathlib.PurePath` or a subclass. When
    `read_interface` is omitted, `SeekIO` is used and `path` must be a `pathlib.Path`.
    """

    path: Final[PathT]
    read_interface: Final[IOReaderAbstract[PathT]]

    @overload
    def __init__(
        self: "FileSystemStream[pathlib.Path]",
        path: pathlib.Path,
        read_interface: None = None,
    ) -> None: ...

    @overload
    def __init__[ReaderPathT: pathlib.PurePath](
        self: "FileSystemStream[ReaderPathT]",
        path: ReaderPathT,
        read_interface: IOReaderAbstract[ReaderPathT],
    ) -> None: ...

    def __init__[ReaderPathT: pathlib.PurePath](
        self: "FileSystemStream[ReaderPathT]",
        path: ReaderPathT,
        read_interface: IOReaderAbstract[ReaderPathT] | None = None,
    ) -> None:
        """Read and write a stream of NumPy arrays in a directory of shard files.

        Args:
            path (pathlib.PurePath): Directory that contains the stream shard files. Its type matches
                `read_interface`. Omitting the reader requires a `pathlib.Path`.
            read_interface (IOReaderAbstract[pathlib.PurePath] | None): Byte-range reader. Defaults to
                `SeekIO`.
        """
        self.path = path
        if read_interface is None:
            if not _is_path_stream(self):
                raise TypeError("path must be a pathlib.Path when read_interface is omitted")
            self.read_interface = SeekIO()
        else:
            self.read_interface = read_interface

    def _standarize_dtype[DType: np.generic](self, dtype: np.dtype[DType]) -> np.dtype[DType]:
        return dtype.newbyteorder("<")

    def sync_reader[DType: np.generic](self, dtype: np.dtype[DType]) -> "_FileSystemStreamSyncReader[DType]":
        """Returns a reader that can access data in the entire dataset.

        Args:
            dtype (np.dtype[DType]): the dtype (possibly structured) that data is stored with.

        Returns:
            _FileSystemStreamSyncReader[DType]: Reader object to synchronously read structured data.
        """
        return _FileSystemStreamSyncReader(self.path, self._standarize_dtype(dtype), self.read_interface)

    def async_reader[DType: np.generic](
        self, dtype: np.dtype[DType]
    ) -> "_FileSystemStreamAsyncReader[DType]":
        """Returns a reader that can access data in the entire dataset.

        Args:
            dtype (np.dtype[DType]): the dtype (possibly structured) that data is stored with.

        Returns:
            _FileSystemStreamAsyncReader[DType]: Reader object to asynchronously read structured data.
        """
        return _FileSystemStreamAsyncReader(self.path, self._standarize_dtype(dtype), self.read_interface)

    def sync_writer[DType: np.generic](self, dtype: np.dtype[DType]) -> "_FileSystemStreamSyncWriter[DType]":
        """Returns a file writer.

        Args:
            dtype (np.dtype[DType]): the dtype (possibly structured) that data is stored with.

        Returns:
            _FileSystemStreamSyncWriter[DType]: Writer object to synchronously append structured data.
        """
        return _FileSystemStreamSyncWriter(self.path, self._standarize_dtype(dtype))


def _is_path_stream(stream: object) -> TypeIs[FileSystemStream[pathlib.Path]]:
    return isinstance(stream, FileSystemStream) and isinstance(stream.path, pathlib.Path)


@final
class _FileSystemStreamSyncReader[DType: np.generic](SyncContextManager):
    def __init__[PathT: pathlib.PurePath](
        self, path: PathT, dtype: np.dtype[DType], read_interface: IOReaderAbstract[PathT]
    ) -> None:
        self.path = path
        self.dtype = dtype
        self._row_bytes = self.dtype.itemsize
        self._make_reader = read_interface.sync_reader()
        self._reader: dict[int, IOSyncReaderAbstract] = {}
        self._open_locks = defaultdict(threading.Lock)

    def open(self) -> None:
        pass

    def close(self) -> None:
        for reader in self._reader.values():
            reader.close()
        self._reader.clear()

    def _get_reader(self, shard: int) -> IOSyncReaderAbstract:
        reader = self._reader.get(shard)
        if reader is not None:
            return reader

        with self._open_locks[shard]:
            # In case of a race conditon
            reader = self._reader.get(shard)
            if reader is not None:
                return reader

            # create reader
            reader = self._make_reader(self.path / _shard_to_filename(shard))
            reader.open()
            self._reader[shard] = reader
            return reader

    def num_shards(self) -> int:
        return num_shards_sync(self._get_reader)

    def length(self, shard: int) -> int:
        try:
            size = self._get_reader(shard).filesize()
            return size // self._row_bytes
        except FileNotFoundError as err:
            raise IndexError(f"this shard ({shard}) does not exist") from err

    def read(self, shard: int, index_in_shard: int, length: int) -> Vector[DType]:
        if shard < 0:
            raise IndexError("Cannot have negative shards")
        if index_in_shard < 0:
            raise IndexError("Cannot read negative index")

        reader = self._get_reader(shard)
        buffer = reader.read(index_in_shard * self._row_bytes, length * self._row_bytes)
        data = np.frombuffer(buffer, dtype=self.dtype)
        if data.shape[0] != length:
            raise RuntimeError("Unexpected number of elements")
        return data


@final
class _FileSystemStreamAsyncReader[DType: np.generic](AsyncContextManager):
    def __init__[PathT: pathlib.PurePath](
        self, path: PathT, dtype: np.dtype[DType], read_interface: IOReaderAbstract[PathT]
    ) -> None:
        self.path = path
        self.dtype = dtype
        self._row_bytes = self.dtype.itemsize
        self._make_reader = read_interface.async_reader()
        self._reader: dict[int, IOAsyncReaderAbstract] = {}

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        await asyncio.gather(*(reader.close() for reader in self._reader.values()))
        self._reader.clear()

    async def _get_reader(self, shard: int) -> IOAsyncReaderAbstract:
        reader = self._reader.get(shard, None)
        if reader is None:
            reader = self._make_reader(self.path / _shard_to_filename(shard))
            await reader.open()
            self._reader[shard] = reader
        return reader

    async def num_shards(self) -> int:
        return await num_shards_async(self._get_reader)

    async def length(self, shard: int) -> int:
        try:
            size = await (await self._get_reader(shard)).filesize()
            return size // self._row_bytes
        except FileNotFoundError as err:
            raise IndexError(f"this shard ({shard}) does not exist") from err

    async def read(self, shard: int, index_in_shard: int, length: int) -> Vector[DType]:
        if shard < 0:
            raise IndexError("Cannot have negative shards")
        if index_in_shard < 0:
            raise IndexError("Cannot read negative index")

        reader = await self._get_reader(shard)
        buffer = await reader.read(index_in_shard * self._row_bytes, length * self._row_bytes)
        data = np.frombuffer(buffer, dtype=self.dtype)
        if data.shape[0] != length:
            raise IndexError("Read index out of file range")
        return data


@final
class _FileSystemStreamSyncWriter[DType: np.generic](SyncContextManager):
    def __init__(self, path: pathlib.PurePath, dtype: np.dtype[DType]) -> None:
        self.path = path
        self.dtype = dtype
        self._dtype_ge = dtype.newbyteorder(">")
        self._row_bytes = self.dtype.itemsize

        self._open = False
        self._fp: dict[int, BinaryIO] = {}
        self._bytes_written: dict[int, int] = {}

    def _get_fp(self, shard: int) -> BinaryIO:
        if self._open is None:
            raise ValueError("writer is not open")
        if shard not in self._fp:
            self._fp[shard] = open(self.path / _shard_to_filename(shard), "wb")  # noqa: SIM115
            self._bytes_written[shard] = 0
        return self._fp[shard]

    def length(self, shard: int) -> int:
        return self._bytes_written.get(shard, 0) // self._row_bytes

    def open(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        self._open = True

    def close(self) -> None:
        for fp in self._fp.values():
            fp.close()
        self._fp = {}
        self._open = False

    def append(self, shard: int, arr: Vector[DType]) -> None:
        if shard < 0:
            raise IndexError("Cannot have negative shards")
        if arr.dtype == self._dtype_ge:
            arr = arr.view(self.dtype).byteswap()
        if arr.dtype != self.dtype:
            raise ValueError(f"unexpected dtype, expected ({self.dtype}) was ({arr.dtype})")
        if arr.ndim != 1:
            raise ValueError("numpy array must be 1-D, use structured arrays for multiple columns")

        content = arr.tobytes(order="C")
        self._bytes_written[shard] = self._bytes_written.get(shard, 0) + len(content)
        self._get_fp(shard).write(content)
