from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict
import shutil
import struct
import asyncio
import pathlib

import numpy as np
import pytest

from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.storage import DocumentStorage
from guideloom.types import DocumentStorageFormat, IOReaderAbstract, StorageBufferTypes, UserBufferTypes


#
# The ConceptStorageFormat applied thoughout this test, fairly basic. But should cover everything.
#
class InDocumentData(TypedDict):
    doc_id: int
    data: str


class OutDocumentData(TypedDict):
    shard: int
    doc_id: int
    data: str


class ConceptStorageFormat(DocumentStorageFormat[InDocumentData, OutDocumentData]):
    def __init__(self) -> None:
        self._prefix_struct = struct.Struct(">I")
        super().__init__()

    def pack(self, data: InDocumentData) -> UserBufferTypes:
        content = data["data"].encode()
        prefix = self._prefix_struct.pack(data["doc_id"])
        return prefix + content

    def unpack(self, encoded_data: StorageBufferTypes, shard: int) -> OutDocumentData:
        (doc_id,) = self._prefix_struct.unpack(encoded_data[: self._prefix_struct.size])
        content_encoded = encoded_data[self._prefix_struct.size :]
        if isinstance(content_encoded, memoryview):
            content_encoded = content_encoded.tobytes()
        return {"doc_id": doc_id, "shard": shard, "data": content_encoded.decode()}


#
# I/O interface
#
@dataclass
class IoDef:
    name: str
    fs_read_interface: IOReaderAbstract[pathlib.Path] | None = None

    def create(self, path: pathlib.Path) -> DocumentStorage[InDocumentData, OutDocumentData]:
        formatter = ConceptStorageFormat()
        return DocumentStorage(path, formatter, read_interface=self.fs_read_interface)


io_interfaces: list[IoDef] = [
    IoDef("fs-default"),
    IoDef("fs-mmap", MmapIO()),
    IoDef("fs-seek", SeekIO()),
]
if PosixPreadIO.is_supported(advice=False):
    io_interfaces.append(IoDef("fs-pread", PosixPreadIO(advice=False)))
if PosixPreadIO.is_supported(advice=True):
    io_interfaces.append(IoDef("fs-advice", PosixPreadIO(advice=True)))


#
# For basic dtypes (not structured dtypes)
#
@pytest.fixture
def documents() -> Sequence[InDocumentData]:
    return [
        {"doc_id": 1, "data": "A"},
        {"doc_id": 2, "data": "B"},
        {"doc_id": 3, "data": "C"},
        {"doc_id": 4, "data": "D"},
        {"doc_id": 5, "data": "E"},
        {"doc_id": 6, "data": "F"},
        {"doc_id": 7, "data": "G"},
        {"doc_id": 8, "data": "H"},
        {"doc_id": 9, "data": "I"},
        {"doc_id": 10, "data": "J"},
        {"doc_id": 11, "data": "K"},
        {"doc_id": 12, "data": "L"},
        {"doc_id": 13, "data": "M"},
        {"doc_id": 14, "data": "N"},
        {"doc_id": 15, "data": "O"},
        {"doc_id": 16, "data": "P"},
        {"doc_id": 17, "data": "Q"},
        {"doc_id": 18, "data": "R"},
        {"doc_id": 19, "data": "S"},
        {"doc_id": 20, "data": "T"},
    ]


def _assert_documents_match(
    actual: Sequence[OutDocumentData], expected: Sequence[InDocumentData], expected_shard: int | None = None
) -> None:
    for actual_item, expected_item in zip(actual, expected, strict=True):
        assert actual_item["doc_id"] == expected_item["doc_id"]
        assert actual_item["data"] == expected_item["data"]
        assert actual_item["shard"] >= 0
        if expected_shard is not None:
            assert actual_item["shard"] == expected_shard


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_single_shard_sync(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_single_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_shard_sync")

    with storage.sync_writer() as writer:
        for doc in documents:
            writer.append(0, doc)
        assert writer.length(0) == len(documents)

    with storage.sync_reader() as reader:
        actual = [reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual, documents, expected_shard=0)
        assert reader.length(0) == len(documents)
        assert reader.total_length == len(documents)
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_single_shard_async(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_single_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_shard_async")

    with storage.sync_writer() as writer:
        for doc in documents:
            writer.append(0, doc)
        assert writer.length(0) == len(documents)

    async with storage.async_reader() as reader:
        actual = [await reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual, documents, expected_shard=0)
        assert reader.length(0) == len(documents)
        assert reader.total_length == len(documents)
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_single_shard_async_parallel(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_single_shard_async_parallel", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_shard_async_parallel")

    with storage.sync_writer() as writer:
        for doc in documents:
            writer.append(0, doc)
        assert writer.length(0) == len(documents)

    async with storage.async_reader() as reader:
        actual = await asyncio.gather(*(reader.read(i) for i in range(len(documents))))
        _assert_documents_match(actual, documents, expected_shard=0)
        assert reader.length(0) == len(documents)
        assert reader.total_length == len(documents)
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_two_shard_sync(tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_sync")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    with storage.sync_reader() as reader:
        actual = [reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_shard_async(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_async")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    async with storage.async_reader() as reader:
        actual = [await reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_shard_async_parallel(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_async_parallel", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_async_parallel")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    async with storage.async_reader() as reader:
        actual = await asyncio.gather(*(reader.read(i) for i in range(len(documents))))
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_two_shard_reverse_order_sync(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_sync")

    with storage.sync_writer() as writer:
        for doc in documents[10:]:
            writer.append(1, doc)
        for doc in documents[:10]:
            writer.append(0, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    with storage.sync_reader() as reader:
        actual = [reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_shard_reverse_order_async(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_async")

    with storage.sync_writer() as writer:
        for doc in documents[10:]:
            writer.append(1, doc)
        for doc in documents[:10]:
            writer.append(0, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    async with storage.async_reader() as reader:
        actual = [await reader.read(i) for i in range(len(documents))]
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_shard_reverse_order_async_parallel(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_async_parallel", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_async_parallel")

    with storage.sync_writer() as writer:
        for doc in documents[10:]:
            writer.append(1, doc)
        for doc in documents[:10]:
            writer.append(0, doc)
        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])

    async with storage.async_reader() as reader:
        actual = await asyncio.gather(*(reader.read(i) for i in range(len(documents))))
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)

        assert writer.length(0) == len(documents[:10])
        assert writer.length(1) == len(documents[10:])
        assert reader.total_length == len(documents)
        assert reader.num_shards == 2


#
# Edge cases
#
@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_empty_shard_sync(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_empty_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_empty_shard_sync")

    with storage.sync_writer() as reader:
        pass

    with storage.sync_reader() as reader:
        assert reader.num_shards == 0
        assert reader.total_length == 0

        with pytest.raises(IndexError, match="out of range"):
            reader.length(0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_empty_shard_async(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_empty_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_empty_shard_async")

    with storage.sync_writer() as reader:
        pass

    async with storage.async_reader() as reader:
        assert reader.num_shards == 0
        assert reader.total_length == 0

        with pytest.raises(IndexError, match="out of range"):
            reader.length(0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_out_of_bounds_data_sync(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_data_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_data_sync")

    with storage.sync_writer() as writer:
        writer.append(0, documents[0])

    with storage.sync_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=1)

        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=2)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_out_of_bounds_data_async(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_data_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_data_async")

    with storage.sync_writer() as writer:
        writer.append(0, documents[0])

    async with storage.async_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=1)

        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=2)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_single_shard_sync_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_single_shard_sync_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_shard_sync_read_many")

    with storage.sync_writer() as writer:
        for doc in documents:
            writer.append(0, doc)

    with storage.sync_reader() as reader:
        actual = reader.read_many(np.arange(len(documents), dtype=np.int64))
        _assert_documents_match(actual, documents, expected_shard=0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_single_shard_async_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_single_shard_async_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_shard_async_read_many")

    with storage.sync_writer() as writer:
        for doc in documents:
            writer.append(0, doc)

    async with storage.async_reader() as reader:
        actual = await reader.read_many(np.arange(len(documents), dtype=np.int64))
        _assert_documents_match(actual, documents, expected_shard=0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_two_shard_sync_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_sync_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_sync_read_many")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)

    with storage.sync_reader() as reader:
        actual = reader.read_many(np.arange(len(documents), dtype=np.int64))
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_shard_async_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_two_shard_async_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_shard_async_read_many")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)

    async with storage.async_reader() as reader:
        actual = await reader.read_many(np.arange(len(documents), dtype=np.int64))
        _assert_documents_match(actual[:10], documents[:10], expected_shard=0)
        _assert_documents_match(actual[10:], documents[10:], expected_shard=1)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_gapped_indexes_sync_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_gapped_indexes_sync_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_gapped_indexes_sync_read_many")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)

    indexes = np.array([15, 0, 12, 3, 19], dtype=np.int64)
    expected = [documents[i] for i in indexes]
    with storage.sync_reader() as reader:
        actual = reader.read_many(indexes)
        _assert_documents_match(
            [item for item, idx in zip(actual, indexes, strict=True) if idx < 10],
            [item for item, idx in zip(expected, indexes, strict=True) if idx < 10],
            expected_shard=0,
        )
        _assert_documents_match(
            [item for item, idx in zip(actual, indexes, strict=True) if idx >= 10],
            [item for item, idx in zip(expected, indexes, strict=True) if idx >= 10],
            expected_shard=1,
        )


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_gapped_indexes_async_read_many(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_gapped_indexes_async_read_many", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_gapped_indexes_async_read_many")

    with storage.sync_writer() as writer:
        for doc in documents[:10]:
            writer.append(0, doc)
        for doc in documents[10:]:
            writer.append(1, doc)

    indexes = np.array([15, 0, 12, 3, 19], dtype=np.int64)
    expected = [documents[i] for i in indexes]
    async with storage.async_reader() as reader:
        actual = await reader.read_many(indexes)
        _assert_documents_match(
            [item for item, idx in zip(actual, indexes, strict=True) if idx < 10],
            [item for item, idx in zip(expected, indexes, strict=True) if idx < 10],
            expected_shard=0,
        )
        _assert_documents_match(
            [item for item, idx in zip(actual, indexes, strict=True) if idx >= 10],
            [item for item, idx in zip(expected, indexes, strict=True) if idx >= 10],
            expected_shard=1,
        )


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_empty_read_many_sync(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_empty_read_many_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_empty_read_many_sync")

    with storage.sync_writer():
        pass

    with storage.sync_reader() as reader:
        assert reader.read_many(np.arange(0, dtype=np.int64)) == []


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_empty_read_many_async(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_empty_read_many_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_empty_read_many_async")

    with storage.sync_writer():
        pass

    async with storage.async_reader() as reader:
        assert await reader.read_many(np.arange(0, dtype=np.int64)) == []


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_out_of_bounds_read_many_sync(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_read_many_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_read_many_sync")

    with storage.sync_writer() as writer:
        writer.append(0, documents[0])

    with storage.sync_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            reader.read_many(np.array([0, 1], dtype=np.int64))

        with pytest.raises(IndexError, match="outside of size"):
            reader.read_many(np.array([-1], dtype=np.int64))


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_out_of_bounds_read_many_async(
    tmp_path: pathlib.Path, io_def: IoDef, documents: Sequence[InDocumentData]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_read_many_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_read_many_async")

    with storage.sync_writer() as writer:
        writer.append(0, documents[0])

    async with storage.async_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            await reader.read_many(np.array([0, 1], dtype=np.int64))

        with pytest.raises(IndexError, match="outside of size"):
            await reader.read_many(np.array([-1], dtype=np.int64))
