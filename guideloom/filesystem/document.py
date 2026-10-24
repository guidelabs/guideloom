from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
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
    StorageBufferTypes,
    SyncContextManager,
    UserBufferTypes,
    Vector,
)
from ._document_compute_requests import (
    compute_data_requests,
    compute_index_requests,
    concat_fast,
    index_data_dtype,
    validate_document_requests,
)
from ._shard_count import num_shards_async, num_shards_sync

type Scalar[DType: np.generic] = np.ndarray[tuple[()], np.dtype[DType]]


def _shard_to_index_filename(shard: int) -> str:
    return f"{shard:07d}.out_of_band.index"


def _shard_to_data_filename(shard: int) -> str:
    return f"{shard:07d}.out_of_band.bin"


@final
class FileSystemDocument[PathT: pathlib.PurePath = pathlib.Path]:
    """Writes and reads document data.

    `path` and `read_interface` are immutable and share `PathT`. Built-in backends require a
    `pathlib.Path`. A custom backend can require `pathlib.PurePath` or a subclass. When
    `read_interface` is omitted, `SeekIO` is used and `path` must be a `pathlib.Path`.
    """

    path: Final[PathT]
    read_interface: Final[IOReaderAbstract[PathT]]

    @overload
    def __init__(
        self: "FileSystemDocument[pathlib.Path]",
        path: pathlib.Path,
        read_interface: None = None,
    ) -> None: ...

    @overload
    def __init__[ReaderPathT: pathlib.PurePath](
        self: "FileSystemDocument[ReaderPathT]",
        path: ReaderPathT,
        read_interface: IOReaderAbstract[ReaderPathT],
    ) -> None: ...

    def __init__[ReaderPathT: pathlib.PurePath](
        self: "FileSystemDocument[ReaderPathT]",
        path: ReaderPathT,
        read_interface: IOReaderAbstract[ReaderPathT] | None = None,
    ) -> None:
        """Writes and reads document data.

        Args:
            path (pathlib.PurePath): Directory that contains the document shard files. Its type matches
                `read_interface`. Omitting the reader requires a `pathlib.Path`.
            read_interface (IOReaderAbstract[pathlib.PurePath] | None): Byte-range reader. Defaults to
                `SeekIO`.
        """
        self.path = path
        if read_interface is None:
            if not _is_path_document(self):
                raise TypeError("path must be a pathlib.Path when read_interface is omitted")
            self.read_interface = SeekIO()
        else:
            self.read_interface = read_interface

    def sync_reader(self) -> "_FileSystemDocumentSyncReader":
        """Returns a file reader.

        Returns:
            _FileSystemDocumentSyncReader: Reader object to synchronously read data.
        """
        return _FileSystemDocumentSyncReader(self.path, self.read_interface)

    def async_reader(self) -> "_FileSystemDocumentAsyncReader":
        """Returns a file reader.

        Returns:
            _FileSystemDocumentAsyncReader: Reader object to asynchronously read data.
        """
        return _FileSystemDocumentAsyncReader(self.path, self.read_interface)

    def sync_writer(self) -> "_FileSystemDocumentSyncWriter":
        """Returns a file writer.

        Returns:
            _FileSystemDocumentSyncWriter: Writer object to synchronously append data.
        """
        return _FileSystemDocumentSyncWriter(self.path)


def _is_path_document(document: object) -> TypeIs[FileSystemDocument[pathlib.Path]]:
    return isinstance(document, FileSystemDocument) and isinstance(document.path, pathlib.Path)


@dataclass(slots=True, eq=False)
class IndexDataReader[T]:
    index: T
    data: T
    shard: int


def _validate_request(shard: int, index: int) -> None:
    if shard < 0:
        raise IndexError("Cannot have negative shards")
    if index < 0:
        raise IndexError("Cannot read negative index")


def _validate_requests(requests: Vector[np.void]) -> None:
    status = validate_document_requests(requests)
    if status == 1:
        raise IndexError("Cannot have negative shards")
    if status == 2:
        raise IndexError("Cannot read negative index")


def _unpack_data(unpack_info: Vector[np.void], datas: Sequence[StorageBufferTypes]) -> Sequence[memoryview]:
    data_views = [memoryview(data) for data in datas]
    return [
        data_views[request][start:end]
        for request, start, end in zip(
            unpack_info["request"].tolist(),
            unpack_info["start"].tolist(),
            unpack_info["end"].tolist(),
            strict=True,
        )
    ]


@final
class _FileSystemDocumentSyncReader(SyncContextManager):
    def __init__[PathT: pathlib.PurePath](self, path: PathT, read_interface: IOReaderAbstract[PathT]) -> None:
        self.path = path
        self.index_dtype = index_data_dtype
        self.index_dtype_bytes = index_data_dtype.itemsize
        self._make_reader = read_interface.sync_reader()
        self._reader: dict[int, IndexDataReader[IOSyncReaderAbstract]] = {}
        self._open_locks = defaultdict(threading.Lock)

    def open(self) -> None:
        pass

    def close(self) -> None:
        for reader in self._reader.values():
            reader.index.close()
            reader.data.close()

    def _get_reader(self, shard: int) -> IndexDataReader[IOSyncReaderAbstract]:
        reader = self._reader.get(shard)
        if reader is not None:
            return reader

        with self._open_locks[shard]:
            # In case of a race conditon
            reader = self._reader.get(shard)
            if reader is not None:
                return reader

            # create reader
            reader = IndexDataReader(
                index=self._make_reader(self.path / _shard_to_index_filename(shard)),
                data=self._make_reader(self.path / _shard_to_data_filename(shard)),
                shard=shard,
            )
            reader.index.open()
            reader.data.open()
            self._reader[shard] = reader
            return reader

    def num_shards(self) -> int:
        return num_shards_sync(self._get_reader)

    def length(self, shard: int) -> int:
        try:
            size = self._get_reader(shard).index.filesize()
            return (size // self.index_dtype_bytes) - 1
        except FileNotFoundError as err:
            raise IndexError(f"this shard ({shard}) does not exist") from err

    def _read_index(
        self, reader: IndexDataReader[IOSyncReaderAbstract], index: int, length: int
    ) -> Vector[np.int64]:
        data_index_buffer = reader.index.read(index * self.index_dtype_bytes, length * self.index_dtype_bytes)
        data_index: Vector[np.int64] = np.frombuffer(data_index_buffer, dtype=self.index_dtype)  # type: ignore
        if data_index.shape[0] != length:
            raise RuntimeError("Unexpected number of elements")
        return data_index

    def get(self, shard: int, index: int) -> StorageBufferTypes:
        _validate_request(shard, index)
        reader = self._get_reader(shard)
        data_index = self._read_index(reader, index, 2)
        return reader.data.read(data_index[0], data_index[1] - data_index[0])

    def get_many(self, requests: Vector[np.void]) -> Sequence[memoryview]:
        if requests.size == 0:
            return []

        # prepare
        _validate_requests(requests)
        index_reqs = compute_index_requests(requests)
        readers = {index_req["shard"]: self._get_reader(index_req["shard"]) for index_req in index_reqs}

        # get indexes
        data_indexes = concat_fast(
            [
                self._read_index(readers[index_req["shard"]], index_req["index"], index_req["length"])
                for index_req in index_reqs
            ],
            dtype=self.index_dtype,
        )

        # get data
        data_reqs, data_unpack_info = compute_data_requests(index_reqs, data_indexes)
        datas = [
            readers[data_req["shard"]].data.read(data_req["offset"], data_req["length"])
            for data_req in data_reqs
        ]

        # unpack data
        return _unpack_data(data_unpack_info, datas)


@final
class _FileSystemDocumentAsyncReader(AsyncContextManager):
    def __init__[PathT: pathlib.PurePath](self, path: PathT, read_interface: IOReaderAbstract[PathT]) -> None:
        self.path = path
        self.index_dtype = index_data_dtype
        self.index_dtype_bytes = self.index_dtype.itemsize
        self._make_reader = read_interface.async_reader()
        self._reader: dict[int, IndexDataReader[IOAsyncReaderAbstract]] = {}

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        await asyncio.gather(
            *(reader.index.close() for reader in self._reader.values()),
            *(reader.data.close() for reader in self._reader.values()),
        )

    async def _get_reader(self, shard: int) -> IndexDataReader[IOAsyncReaderAbstract]:
        reader = self._reader.get(shard, None)
        if reader is None:
            reader = IndexDataReader(
                index=self._make_reader(self.path / _shard_to_index_filename(shard)),
                data=self._make_reader(self.path / _shard_to_data_filename(shard)),
                shard=shard,
            )
            await asyncio.gather(reader.data.open(), reader.index.open())
            self._reader[shard] = reader

        return reader

    async def num_shards(self) -> int:
        return await num_shards_async(self._get_reader)

    async def length(self, shard: int) -> int:
        try:
            size = await (await self._get_reader(shard)).index.filesize()
            return (size // self.index_dtype_bytes) - 1
        except FileNotFoundError as err:
            raise IndexError(f"this shard ({shard}) does not exist") from err

    async def _read_index(
        self, reader: IndexDataReader[IOAsyncReaderAbstract], index: int, length: int
    ) -> Vector[np.int64]:
        data_index_buffer = await reader.index.read(
            index * self.index_dtype_bytes, length * self.index_dtype_bytes
        )
        data_index: Vector[np.int64] = np.frombuffer(data_index_buffer, dtype=self.index_dtype)  # type: ignore
        if data_index.shape[0] != length:
            raise IndexError("Read index out of file range")
        return data_index

    async def get(self, shard: int, index: int) -> StorageBufferTypes:
        _validate_request(shard, index)
        reader = await self._get_reader(shard)
        data_index = await self._read_index(reader, index, 2)
        return await reader.data.read(data_index[0], data_index[1] - data_index[0])

    async def get_many(self, requests: Vector[np.void]) -> Sequence[memoryview]:
        if requests.size == 0:
            return []

        # prepare
        _validate_requests(requests)
        index_reqs = compute_index_requests(requests)
        readers = {
            reader.shard: reader
            for reader in await asyncio.gather(
                *(self._get_reader(index_req["shard"]) for index_req in index_reqs)
            )
        }

        # get indexes
        data_indexes = concat_fast(
            await asyncio.gather(
                *(
                    self._read_index(readers[index_req["shard"]], index_req["index"], index_req["length"])
                    for index_req in index_reqs
                )
            ),
            dtype=self.index_dtype,
        )

        # get data
        data_reqs, data_unpack_info = compute_data_requests(index_reqs, data_indexes)
        datas = await asyncio.gather(
            *(
                readers[data_req["shard"]].data.read(data_req["offset"], data_req["length"])
                for data_req in data_reqs
            )
        )

        # unpack data
        return _unpack_data(data_unpack_info, datas)


@dataclass(slots=True, eq=False)
class IndexDataFilePointer:
    index: BinaryIO
    data: BinaryIO
    num_obs: int = 0
    data_tell: int = 0


@final
class _FileSystemDocumentSyncWriter(SyncContextManager):
    def __init__(self, path: pathlib.PurePath) -> None:
        self.path = path
        self.index_dtype = index_data_dtype
        self.index_dtype_bytes = self.index_dtype.itemsize
        self.padding_value = np.array((0, -1), dtype=self.index_dtype)

        self._fp: dict[int, IndexDataFilePointer] = {}
        self._open = False

    def _get_fp(self, shard: int) -> IndexDataFilePointer:
        if not self._open:
            raise ValueError("writer is not open")
        if shard not in self._fp:
            self._fp[shard] = IndexDataFilePointer(
                index=open(self.path / _shard_to_index_filename(shard), "wb"),  # noqa: SIM115
                data=open(self.path / _shard_to_data_filename(shard), "wb"),  # noqa: SIM115
            )
            self._fp[shard].index.write(np.array(self._fp[shard].data_tell, dtype=self.index_dtype).tobytes())
        return self._fp[shard]

    def length(self, shard: int) -> int:
        return self._get_fp(shard).num_obs

    def open(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        self._open = True

    def close(self) -> None:
        for fp in self._fp.values():
            fp.index.close()
            fp.data.close()
        self._fp = {}
        self._open = False

    def append(self, shard: int, value: UserBufferTypes) -> None:
        if shard < 0:
            raise IndexError("Cannot have negative shards")
        fp = self._get_fp(shard)
        fp.data.write(value)
        fp.data_tell += len(value)

        # fast path, at the correct file position, just append
        index_item = np.array(fp.data_tell, dtype=self.index_dtype)
        fp.index.write(index_item.tobytes())
        fp.num_obs += 1
