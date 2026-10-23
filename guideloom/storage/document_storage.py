from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, final, overload
import asyncio
import pathlib

import numpy as np

from ..filesystem import FileSystemDocument
from ..io import SeekIO
from ..types import (
    DocumentStorageAbstract,
    DocumentStorageAsyncReaderAbstract,
    DocumentStorageFormat,
    DocumentStorageSyncReaderAbstract,
    DocumentStorageSyncWriterAbstract,
    IOReaderAbstract,
    Vector,
)

_DOCUMENT_REQ_DTYPE = np.dtype(
    [
        ("shard", np.int64),
        ("shard_index", np.int32),
    ],
    align=True,
)


@dataclass(slots=True, eq=False)
class _DocumentReq:
    shard: int
    shard_index: int


@final
class DocumentStorage[InDataT, OutDataT](DocumentStorageAbstract[InDataT, OutDataT]):
    @overload
    def __init__(
        self,
        storage_dir: pathlib.Path,
        storage_format: DocumentStorageFormat[InDataT, OutDataT],
        read_interface: None = None,
    ) -> None: ...

    @overload
    def __init__[PathT: pathlib.PurePath](
        self,
        storage_dir: PathT,
        storage_format: DocumentStorageFormat[InDataT, OutDataT],
        read_interface: IOReaderAbstract[PathT],
    ) -> None: ...

    def __init__[PathT: pathlib.PurePath](
        self,
        storage_dir: PathT,
        storage_format: DocumentStorageFormat[InDataT, OutDataT],
        read_interface: IOReaderAbstract[PathT] | None = None,
    ) -> None:
        """Create document storage for a directory of shard files.

        `storage_dir` and `read_interface` share `PathT`. Built-in backends require a `pathlib.Path`.
        A custom backend can require `pathlib.PurePath` or a subclass. When `read_interface` is
        omitted, `SeekIO` is used and `storage_dir` must be a `pathlib.Path`.

        Args:
            storage_dir (PathT): Directory that contains the shard files.
            storage_format (DocumentStorageFormat[InDataT, OutDataT]): Encodes and decodes each document.
            read_interface (IOReaderAbstract[PathT] | None): Byte-range reader. Defaults to `SeekIO`.
        """
        if read_interface is None:
            if not isinstance(storage_dir, pathlib.Path):
                raise TypeError("storage_dir must be a pathlib.Path when read_interface is omitted")
            read_interface = cast(IOReaderAbstract[PathT], SeekIO())
        self._document_io = FileSystemDocument(storage_dir, read_interface)
        self.storage_format = storage_format

    def sync_writer(self) -> "_DocumentStorageSyncWriter[InDataT]":
        """Return a synchronous writer for this storage.

        Use the writer as a context manager. See `DocumentStorageSyncWriterAbstract`.

        Returns:
            DocumentStorageSyncWriterAbstract[InDataT]: Writer that appends documents to shards.
        """
        return _DocumentStorageSyncWriter(self.storage_format, self._document_io)

    def sync_reader(self) -> "_DocumentStorageSyncReader[OutDataT]":
        """Return a synchronous reader over every shard in this storage.

        Use the reader as a context manager. See `DocumentStorageSyncReaderAbstract`.

        Returns:
            DocumentStorageSyncReaderAbstract[OutDataT]: Reader over the stored documents.
        """
        return _DocumentStorageSyncReader(self.storage_format, self._document_io)

    def async_reader(self) -> "_DocumentStorageAsyncReader[OutDataT]":
        """Return an asynchronous reader over every shard in this storage.

        Use the reader as a context manager. See `DocumentStorageAsyncReaderAbstract`.

        Returns:
            DocumentStorageAsyncReaderAbstract[OutDataT]: Reader over the stored documents.
        """
        return _DocumentStorageAsyncReader(self.storage_format, self._document_io)


@final
class _DocumentStorageSyncWriter[InDataT](DocumentStorageSyncWriterAbstract[InDataT]):
    def __init__[PathT: pathlib.PurePath](
        self, storage_format: DocumentStorageFormat[InDataT, Any], document_io: FileSystemDocument[PathT]
    ) -> None:
        self.storage_format = storage_format
        self.document_writer = document_io.sync_writer()

        self._items_written: dict[int, int] = {}

    def open(self) -> None:
        self._items_written = {}
        self.document_writer.open()

    def close(self) -> None:
        self.document_writer.close()

    def append(self, shard: int, data: InDataT) -> None:
        data_encoded = self.storage_format.pack(data)
        if not isinstance(data_encoded, bytes | memoryview):
            raise TypeError("encoded data must be bytes-like object")

        self.document_writer.append(shard, data_encoded)
        self._items_written[shard] = self._items_written.get(shard, 0) + 1

    def length(self, shard: int) -> int:
        return self._items_written[shard]


class _DocumentStorageShareReader:
    def __init__(self) -> None:
        self._open = False
        self._num_shards: int = 0
        self._total_length: int = 0
        self._shard_length: Vector[np.int64] = np.zeros((0,), dtype=np.int64)
        self._offset_index: Vector[np.int64] = np.zeros((0,), dtype=np.int64)

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

    def _compute_request(self, global_index: int) -> _DocumentReq:
        if global_index >= self._total_length:
            raise IndexError(f"requested index ({global_index}) is outside of size ({self._total_length})")

        # get the shard where the data is stored, and the index into the shard
        shard = np.searchsorted(self._offset_index, global_index, side="right") - 1
        index_in_shard: int = (global_index - self._offset_index[shard]).item()
        assert index_in_shard >= 0
        return _DocumentReq(shard.item(), index_in_shard)

    def _compute_requests(self, global_indexes: Vector[np.int64]) -> Vector[np.void]:
        if global_indexes.size == 0:
            return np.empty((0,), dtype=_DOCUMENT_REQ_DTYPE)

        out_of_range = (global_indexes < 0) | (global_indexes >= self._total_length)
        if np.any(out_of_range):
            global_index = global_indexes[out_of_range].item(0)
            raise IndexError(f"requested index ({global_index}) is outside of size ({self._total_length})")

        shards = np.searchsorted(self._offset_index, global_indexes, side="right") - 1
        shard_indexes = global_indexes - self._offset_index[shards]

        requests = np.empty(global_indexes.shape[0], dtype=_DOCUMENT_REQ_DTYPE)
        requests["shard"] = shards
        requests["shard_index"] = shard_indexes
        return requests


@final
class _DocumentStorageSyncReader[OutDataT](
    _DocumentStorageShareReader, DocumentStorageSyncReaderAbstract[OutDataT]
):
    def __init__[PathT: pathlib.PurePath](
        self, storage_format: DocumentStorageFormat[Any, OutDataT], document_io: FileSystemDocument[PathT]
    ) -> None:
        self.storage_format = storage_format
        self.document_reader = document_io.sync_reader()
        super().__init__()

    def open(self) -> None:
        self.document_reader.open()
        self._num_shards = self.document_reader.num_shards()
        self._set_length([self.document_reader.length(shard) for shard in range(self._num_shards)])
        self._open = True

    def close(self) -> None:
        self.document_reader.close()
        self._open = False

    def read(self, global_index: int) -> OutDataT:
        req = self._compute_request(global_index)
        data_encoded = self.document_reader.get(req.shard, req.shard_index)
        return self.storage_format.unpack(data_encoded, req.shard)

    def read_many(self, global_indexes: Vector[np.int64]) -> Sequence[OutDataT]:
        reqs = self._compute_requests(global_indexes)
        encoded = self.document_reader.get_many(reqs)
        return [
            self.storage_format.unpack(data, int(req["shard"]))
            for data, req in zip(encoded, reqs, strict=True)
        ]


@final
class _DocumentStorageAsyncReader[OutDataT](
    _DocumentStorageShareReader, DocumentStorageAsyncReaderAbstract[OutDataT]
):
    def __init__[PathT: pathlib.PurePath](
        self, storage_format: DocumentStorageFormat[Any, OutDataT], document_io: FileSystemDocument[PathT]
    ) -> None:
        self.storage_format = storage_format
        self.document_reader = document_io.async_reader()
        super().__init__()

    async def open(self) -> None:
        await self.document_reader.open()
        self._num_shards = await self.document_reader.num_shards()
        self._set_length(
            await asyncio.gather(*(self.document_reader.length(shard) for shard in range(self._num_shards)))
        )
        self._open = True

    async def close(self) -> None:
        await self.document_reader.close()
        self._open = False

    async def read(self, global_index: int) -> OutDataT:
        req = self._compute_request(global_index)
        data_encoded = await self.document_reader.get(req.shard, req.shard_index)
        return self.storage_format.unpack(data_encoded, req.shard)

    async def read_many(self, global_indexes: Vector[np.int64]) -> Sequence[OutDataT]:
        reqs = self._compute_requests(global_indexes)
        encoded = await self.document_reader.get_many(reqs)
        return [
            self.storage_format.unpack(data, int(req["shard"]))
            for data, req in zip(encoded, reqs, strict=True)
        ]
