from collections.abc import Sequence
from typing import Any, cast, final, overload
import asyncio
import pathlib

import numpy as np

from ..filesystem import FileSystemStream
from ..io import SeekIO
from ..types import (
    IOReaderAbstract,
    StreamStorageAbstract,
    StreamStorageAsyncReaderAbstract,
    StreamStorageFormat,
    StreamStorageSyncReaderAbstract,
    StreamStorageSyncWriterAbstract,
    Vector,
)
from ._compute_requests import compute_stream_requests, concat_fast


@final
class StreamStorage[InDataT, OutDataT, StreamDType: np.generic](StreamStorageAbstract[InDataT, OutDataT]):
    @overload
    def __init__(
        self,
        storage_dir: pathlib.Path,
        storage_format: StreamStorageFormat[InDataT, OutDataT, StreamDType],
        read_interface: None = None,
    ) -> None: ...

    @overload
    def __init__[PathT: pathlib.PurePath](
        self,
        storage_dir: PathT,
        storage_format: StreamStorageFormat[InDataT, OutDataT, StreamDType],
        read_interface: IOReaderAbstract[PathT],
    ) -> None: ...

    def __init__[PathT: pathlib.PurePath](
        self,
        storage_dir: PathT,
        storage_format: StreamStorageFormat[InDataT, OutDataT, StreamDType],
        read_interface: IOReaderAbstract[PathT] | None = None,
    ) -> None:
        """Create stream storage for a directory of shard files.

        `storage_dir` and `read_interface` share `PathT`. Built-in backends require a `pathlib.Path`.
        A custom backend can require `pathlib.PurePath` or a subclass. When `read_interface` is
        omitted, `SeekIO` is used and `storage_dir` must be a `pathlib.Path`.

        Args:
            storage_dir (PathT): Directory that contains the shard files.
            storage_format (StreamStorageFormat[InDataT, OutDataT, StreamDType]): Encodes and decodes
                each stream record.
            read_interface (IOReaderAbstract[PathT] | None): Byte-range reader. Defaults to `SeekIO`.
        """
        if read_interface is None:
            if not isinstance(storage_dir, pathlib.Path):
                raise TypeError("storage_dir must be a pathlib.Path when read_interface is omitted")
            read_interface = cast(IOReaderAbstract[PathT], SeekIO())
        self._stream_io = FileSystemStream(storage_dir, read_interface)
        self.storage_format = storage_format

    def sync_writer(self) -> "_StreamStorageSyncWriter[InDataT, StreamDType]":
        """Return a synchronous writer for this storage.

        Use the writer as a context manager. See `StreamStorageSyncWriterAbstract`.

        Returns:
            StreamStorageSyncWriterAbstract[InDataT]: Writer that appends stream elements to shards.
        """
        return _StreamStorageSyncWriter(self.storage_format, self._stream_io)

    def sync_reader(self) -> "_StreamStorageSyncReader[OutDataT, StreamDType]":
        """Return a synchronous reader over every shard in this storage.

        Use the reader as a context manager. See `StreamStorageSyncReaderAbstract`.

        Returns:
            StreamStorageSyncReaderAbstract[OutDataT]: Reader over the stored stream.
        """
        return _StreamStorageSyncReader(self.storage_format, self._stream_io)

    def async_reader(self) -> "_StreamStorageAsyncReader[OutDataT, StreamDType]":
        """Return an asynchronous reader over every shard in this storage.

        Use the reader as a context manager. See `StreamStorageAsyncReaderAbstract`.

        Returns:
            StreamStorageAsyncReaderAbstract[OutDataT]: Reader over the stored stream.
        """
        return _StreamStorageAsyncReader(self.storage_format, self._stream_io)


@final
class _StreamStorageSyncWriter[InDataT, StreamDType: np.generic](StreamStorageSyncWriterAbstract[InDataT]):
    def __init__[PathT: pathlib.PurePath](
        self,
        storage_format: StreamStorageFormat[InDataT, Any, StreamDType],
        stream_io: FileSystemStream[PathT],
    ) -> None:
        self.storage_format = storage_format
        self.stream_writer = stream_io.sync_writer(storage_format.stream_dtype)
        self._items_written: dict[int, int] = {}

    def open(self) -> None:
        self._items_written = {}
        self.stream_writer.open()

    def close(self) -> None:
        self.stream_writer.close()

    def append(self, shard: int, data: InDataT) -> None:
        stream_data = self.storage_format.pack(data)
        self.stream_writer.append(shard, stream_data)
        self._items_written[shard] = self._items_written.get(shard, 0) + stream_data.size

    def length(self, shard: int) -> int:
        return self._items_written[shard]


class _StreamStorageShareReader[OutDataT, StreamDType: np.generic]:
    def __init__(self, storage_format: StreamStorageFormat[Any, OutDataT, StreamDType]) -> None:
        self.storage_format = storage_format

        self._open = False
        self._num_shards: int = 0
        self._shard_length: Vector[np.int64] = np.zeros((0,), dtype=np.int64)
        self._offset_index: Vector[np.int64] = np.zeros((0,), dtype=np.int64)
        self._total_length: int = 0

    @property
    def num_shards(self) -> int:
        if self._open is None:
            raise RuntimeError("storage was not opened")
        return self._num_shards

    @property
    def total_length(self) -> int:
        if self._open is None:
            raise RuntimeError("storage was not opened")
        return self._total_length

    def length(self, shard: int) -> int:
        if self._open is None:
            raise RuntimeError("storage was not opened")
        if shard >= self.num_shards or shard < 0:
            raise IndexError(f"shard ({shard}) is out of range, there are {self.num_shards} shards.")
        return self._shard_length[shard].item()

    def _set_length(self, lengths: Sequence[int]) -> None:
        self._shard_length = np.array(lengths, dtype=np.int64)  # type: ignore
        self._offset_index = np.concat([[np.int64(0)], np.cumsum(self._shard_length)])  # type: ignore
        self._total_length = self._offset_index[-1].item()

    def _compute_stream_requests(self, global_index: int, global_length: int) -> Vector[np.void]:
        if global_index >= self._total_length:
            raise IndexError(f"requested index ({global_index}) is outside of size ({self._total_length})")
        if global_length < 0:
            raise IndexError(f"requested length ({global_length}) is negative")

        return compute_stream_requests(global_index, global_length, self._offset_index, self._shard_length)


@final
class _StreamStorageSyncReader[OutDataT, StreamDType: np.generic](
    _StreamStorageShareReader[OutDataT, StreamDType], StreamStorageSyncReaderAbstract[OutDataT]
):
    def __init__[PathT: pathlib.PurePath](
        self,
        storage_format: StreamStorageFormat[Any, OutDataT, StreamDType],
        stream_io: FileSystemStream[PathT],
    ) -> None:
        self.storage_format = storage_format
        self.stream_reader = stream_io.sync_reader(storage_format.stream_dtype)

        super().__init__(storage_format)

    def open(self) -> None:
        self.stream_reader.open()

        self._num_shards = self.stream_reader.num_shards()
        self._set_length([self.stream_reader.length(shard) for shard in range(self._num_shards)])
        self._open = True

    def close(self) -> None:
        self.stream_reader.close()
        self._open = False

    def read(self, global_index: int, global_length: int) -> OutDataT:
        stream_reqs = self._compute_stream_requests(global_index, global_length)

        # get stream data
        stream_data_packed_list = [
            self.stream_reader.read(stream_req["shard"], stream_req["shard_offset"], stream_req["length"])
            for stream_req in stream_reqs
        ]

        # unpack data according to users formatter
        stream_data = concat_fast(stream_data_packed_list, self.storage_format.stream_dtype)
        return self.storage_format.unpack(stream_data, stream_reqs["length"], stream_reqs["shard"])


@final
class _StreamStorageAsyncReader[OutDataT, StreamDType: np.generic](
    _StreamStorageShareReader[OutDataT, StreamDType], StreamStorageAsyncReaderAbstract[OutDataT]
):
    def __init__[PathT: pathlib.PurePath](
        self,
        storage_format: StreamStorageFormat[Any, OutDataT, StreamDType],
        stream_io: FileSystemStream[PathT],
    ) -> None:
        self.storage_format = storage_format
        self.stream_reader = stream_io.async_reader(storage_format.stream_dtype)

        super().__init__(storage_format)

    async def open(self) -> None:
        await self.stream_reader.open()

        self._num_shards = await self.stream_reader.num_shards()
        self._set_length(
            await asyncio.gather(*(self.stream_reader.length(shard) for shard in range(self._num_shards)))
        )
        self._open = True

    async def close(self) -> None:
        await self.stream_reader.close()
        self._open = False

    async def read(self, global_index: int, global_length: int) -> OutDataT:
        stream_reqs = self._compute_stream_requests(global_index, global_length)

        # get stream data
        stream_data_packed_list = await asyncio.gather(
            *(
                self.stream_reader.read(stream_req["shard"], stream_req["shard_offset"], stream_req["length"])
                for stream_req in stream_reqs
            )
        )

        # unpack data according to users formatter
        stream_data = concat_fast(stream_data_packed_list, self.storage_format.stream_dtype)
        return self.storage_format.unpack(stream_data, stream_reqs["length"], stream_reqs["shard"])
