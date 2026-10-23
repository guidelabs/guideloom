from collections.abc import Generator, Hashable, Iterator, Sequence
from typing import Any, final
import asyncio
import pathlib

import numpy as np

from ..filesystem import FileSystemDocument, FileSystemStream
from ..io import SeekIO
from ..types import (
    IOReaderAbstract,
    StreamStorageAbstract,
    StreamStorageAsyncReaderAbstract,
    StreamStorageSyncReaderAbstract,
    StreamStorageSyncWriterAbstract,
    StreamWithMetaStorageFormat,
    Vector,
)
from ._compute_requests import compute_document_requests_and_index, compute_stream_requests, concat_fast


class _CaptureGeneratorReturn[T, R]:
    def __init__(self, gen_func: Generator[T, None, R]) -> None:
        self._gen = gen_func
        self._return_val: R | None = None

    def __iter__(self) -> Iterator[T]:
        self._return_val = yield from self._gen
        return

    @property
    def return_value(self) -> R:
        if self._return_val is None:
            raise RuntimeError("Generator not fully consumed yet")
        return self._return_val


@final
class StreamWithMetaStorage[InDataT, OutDataT, StreamDType: np.generic](
    StreamStorageAbstract[InDataT, OutDataT]
):
    def __init__(
        self,
        storage_dir: pathlib.PurePath,
        storage_format: StreamWithMetaStorageFormat[InDataT, OutDataT, StreamDType],
        read_interface: IOReaderAbstract | None = None,
    ) -> None:
        """Create stream-with-metadata storage for a directory of shard files.

        Stream elements and their metadata payloads are both stored under `storage_dir`.

        Args:
            storage_dir (pathlib.PurePath): Directory that contains the shard files.
            storage_format (StreamWithMetaStorageFormat[InDataT, OutDataT, StreamDType]): Encodes and
                decodes each record.
            read_interface (IOReaderAbstract | None): Byte-range reader. Defaults to `SeekIO`.
        """
        read_interface = SeekIO() if read_interface is None else read_interface
        self.storage_format = storage_format
        self._stream_io = FileSystemStream(storage_dir, read_interface=read_interface)
        self._document_io = FileSystemDocument(storage_dir, read_interface=read_interface)

        self._stream_dtype = np.dtype([("index", np.int32), ("user", storage_format.stream_dtype)])

    def sync_writer(self) -> "_StreamWithMetaStorageSyncWriter[InDataT, StreamDType]":
        """Return a synchronous writer for this storage.

        Use the writer as a context manager. See `StreamStorageSyncWriterAbstract`.

        Returns:
            StreamStorageSyncWriterAbstract[InDataT]: Writer that appends stream elements and metadata.
        """
        return _StreamWithMetaStorageSyncWriter(
            self.storage_format, self._stream_io, self._stream_dtype, self._document_io
        )

    def sync_reader(self) -> "_StreamWithMetaStorageSyncReader[OutDataT, StreamDType]":
        """Return a synchronous reader over every shard in this storage.

        Use the reader as a context manager. See `StreamStorageSyncReaderAbstract`.

        Returns:
            StreamStorageSyncReaderAbstract[OutDataT]: Reader over the stored stream and metadata.
        """
        return _StreamWithMetaStorageSyncReader(
            self.storage_format, self._stream_io, self._stream_dtype, self._document_io
        )

    def async_reader(self) -> "_StreamWithMetaStorageAsyncReader[OutDataT, StreamDType]":
        """Return an asynchronous reader over every shard in this storage.

        Use the reader as a context manager. See `StreamStorageAsyncReaderAbstract`.

        Returns:
            StreamStorageAsyncReaderAbstract[OutDataT]: Reader over the stored stream and metadata.
        """
        return _StreamWithMetaStorageAsyncReader(
            self.storage_format, self._stream_io, self._stream_dtype, self._document_io
        )


@final
class _StreamWithMetaStorageSyncWriter[InDataT, StreamDType: np.generic](
    StreamStorageSyncWriterAbstract[InDataT]
):
    def __init__(
        self,
        storage_format: StreamWithMetaStorageFormat[InDataT, Any, StreamDType],
        stream_io: FileSystemStream,
        stream_dtype: np.dtype[np.void],
        document_io: FileSystemDocument,
    ) -> None:
        self.storage_format = storage_format
        self.stream_dtype = stream_dtype
        self.stream_writer = stream_io.sync_writer(stream_dtype)
        self.document_writer = document_io.sync_writer()

        self._curr_document_global_index: dict[int, int] = {}
        self._items_written: dict[int, int] = {}

    def open(self) -> None:
        self._items_written = {}
        self._curr_document_global_index = {}
        self.stream_writer.open()
        self.document_writer.open()

    def close(self) -> None:
        self.stream_writer.close()
        self.document_writer.close()

    def append(self, shard: int, data: InDataT) -> None:
        packing = _CaptureGeneratorReturn(self.storage_format.pack(data))
        num_items = self.storage_format.size(data)
        stream_packed = np.empty(num_items, dtype=self.stream_dtype)

        document_last_key: Hashable = object()
        seen_keys: set[Hashable] = set()
        for stream_index, (document_key, document_value) in enumerate(packing):
            if document_last_key != document_key:
                if document_key in seen_keys:
                    raise ValueError(
                        f"keys must be sequentually unique, got {document_key} but has already been used"
                    )
                seen_keys.add(document_key)

                if not isinstance(document_value, bytes | memoryview):
                    raise ValueError(
                        "document value must be specified and be a bytes-like object"
                        f" for the first unique key, was {document_value}"
                    )

                # its a new key, move the global document index, and write out the value.
                self._curr_document_global_index[shard] = self._curr_document_global_index.get(shard, -1) + 1
                self.document_writer.append(shard, document_value)
                document_last_key = document_key

                if self._curr_document_global_index[shard] >= 2**31:
                    raise ValueError(f"too many unique metadata objects in shard {shard}use less than 2**31")

            stream_packed["index"][stream_index] = self._curr_document_global_index[shard]

        stream_packed["user"] = packing.return_value
        self.stream_writer.append(shard, stream_packed)
        self._items_written[shard] = self._items_written.get(shard, 0) + num_items

    def length(self, shard: int) -> int:
        return self._items_written[shard]


class _StreamWithMetaStorageShareReader[OutDataT, StreamDType: np.generic]:
    def __init__(self, storage_format: StreamWithMetaStorageFormat[Any, OutDataT, StreamDType]) -> None:
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

    def _compute_document_requests_and_index(
        self,
        global_length: int,
        stream_reqs: Vector[np.void],
        stream_data_packed_list: Sequence[Vector[np.void]],
    ) -> tuple[Vector[np.void], Vector[np.int32]]:
        document_indexes = concat_fast(
            [stream_data_packed["index"] for stream_data_packed in stream_data_packed_list],
            dtype=np.dtype(np.int32),
        )
        return compute_document_requests_and_index(global_length, stream_reqs, document_indexes)


@final
class _StreamWithMetaStorageSyncReader[OutDataT, StreamDType: np.generic](
    _StreamWithMetaStorageShareReader[OutDataT, StreamDType], StreamStorageSyncReaderAbstract[OutDataT]
):
    def __init__(
        self,
        storage_format: StreamWithMetaStorageFormat[Any, OutDataT, StreamDType],
        stream_io: FileSystemStream,
        stream_dtype: np.dtype[np.void],
        document_io: FileSystemDocument,
    ) -> None:
        self.stream_dtype = stream_dtype
        self.stream_reader = stream_io.sync_reader(stream_dtype)
        self.document_reader = document_io.sync_reader()

        super().__init__(storage_format)

    def open(self) -> None:
        self.stream_reader.open()
        self.document_reader.open()

        self._num_shards = self.stream_reader.num_shards()
        self._set_length([self.stream_reader.length(shard) for shard in range(self._num_shards)])
        self._open = True

    def close(self) -> None:
        self.stream_reader.close()
        self.document_reader.close()
        self._open = False

    def read(self, global_index: int, global_length: int) -> OutDataT:
        stream_reqs = self._compute_stream_requests(global_index, global_length)

        # get stream data
        stream_data_packed_list = [
            self.stream_reader.read(stream_req["shard"], stream_req["shard_offset"], stream_req["length"])
            for stream_req in stream_reqs
        ]

        document_reqs, document_local_indexes = self._compute_document_requests_and_index(
            global_length, stream_reqs, stream_data_packed_list
        )

        # retrive document data
        document_data = self.document_reader.get_many(document_reqs)

        # unpack data according to users formatter
        stream_data = concat_fast(
            [stream_data_packed["user"] for stream_data_packed in stream_data_packed_list],
            dtype=self.storage_format.stream_dtype,
        )

        return self.storage_format.unpack(
            stream_data, document_reqs["shard"], document_local_indexes, document_data
        )


@final
class _StreamWithMetaStorageAsyncReader[OutDataT, StreamDType: np.generic](
    _StreamWithMetaStorageShareReader[OutDataT, StreamDType], StreamStorageAsyncReaderAbstract[OutDataT]
):
    def __init__(
        self,
        storage_format: StreamWithMetaStorageFormat[Any, OutDataT, StreamDType],
        stream_io: FileSystemStream,
        stream_dtype: np.dtype[np.void],
        document_io: FileSystemDocument,
    ) -> None:
        self.stream_dtype = stream_dtype
        self.stream_reader = stream_io.async_reader(stream_dtype)
        self.document_reader = document_io.async_reader()

        super().__init__(storage_format)

    async def open(self) -> None:
        await self.stream_reader.open()
        await self.document_reader.open()

        self._num_shards = await self.stream_reader.num_shards()
        self._set_length(
            await asyncio.gather(*(self.stream_reader.length(shard) for shard in range(self._num_shards)))
        )
        self._open = True

    async def close(self) -> None:
        await self.stream_reader.close()
        await self.document_reader.close()
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

        document_reqs, document_local_indexes = self._compute_document_requests_and_index(
            global_length, stream_reqs, stream_data_packed_list
        )

        # retrive document data
        document_data = await self.document_reader.get_many(document_reqs)

        # unpack data according to users formatter
        stream_data = concat_fast(
            [stream_data_packed["user"] for stream_data_packed in stream_data_packed_list],
            dtype=self.storage_format.stream_dtype,
        )
        return self.storage_format.unpack(
            stream_data, document_reqs["shard"], document_local_indexes, document_data
        )
