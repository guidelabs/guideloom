from collections.abc import Callable, Generator, Hashable, Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict
import shutil
import asyncio
import pathlib

import numpy as np
import pytest

from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.storage import StreamWithMetaStorage
from guideloom.types import (
    IOReaderAbstract,
    StorageBufferTypes,
    StreamWithMetaStorageFormat,
    UserBufferTypes,
    Vector,
)


#
# The ConceptStorageFormat applied thoughout this test, fairly basic. But should cover everything.
#
class DocumentData[InputDType: np.generic](TypedDict):
    doc_id: int
    document_data: Sequence[int]
    document_id: Vector[np.int32]
    stream_data: Vector[InputDType]


class DocumentItem(TypedDict):
    doc_id: int
    value: int


class StreamingData[InputDType: np.generic](TypedDict):
    document_data: Sequence[DocumentItem]
    document_id: Vector[np.int32]
    stream_data: Vector[InputDType]
    shard: NotRequired[Vector[np.int64]]


class ConceptStorageFormat[DataDType: np.generic](
    StreamWithMetaStorageFormat[DocumentData[DataDType], StreamingData[DataDType], DataDType]
):
    def size(self, data: DocumentData[DataDType]) -> int:
        return data["stream_data"].size

    def pack(
        self, data: DocumentData[DataDType]
    ) -> Generator[tuple[Hashable, UserBufferTypes | None], None, Vector[DataDType]]:
        for document_id in data["document_id"]:
            document_data = data["document_data"][document_id.item()]
            yield ((document_id,), f"{data['doc_id']}-{document_data}".encode())

        return data["stream_data"]

    def unpack(
        self,
        stream_data: Vector[DataDType],
        document_shard: Vector[np.int64],
        document_index: Vector[np.int32],
        document_data: Sequence[StorageBufferTypes],
    ) -> StreamingData[DataDType]:
        document_data_unpack: list[list[str]] = [
            (payload.tobytes() if isinstance(payload, memoryview) else payload).decode().split("-")
            for payload in document_data
        ]
        return {
            "document_data": [
                {"doc_id": int(document_data_unpack[0]), "value": int(document_data_unpack[1])}
                for document_data_unpack in document_data_unpack
            ],
            "shard": document_shard,
            "document_id": document_index,
            "stream_data": stream_data,
        }


#
# I/O interface
#
@dataclass
class IoDef:
    name: str
    fs_read_interface: IOReaderAbstract[pathlib.Path] | None = None

    def create[DType: np.generic](
        self, path: pathlib.Path, stream_dtype: np.dtype[DType]
    ) -> StreamWithMetaStorage[DocumentData[DType], StreamingData[DType], DType]:
        formatter = ConceptStorageFormat(stream_dtype=stream_dtype)
        return StreamWithMetaStorage(path, formatter, read_interface=self.fs_read_interface)


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
def doc_first() -> DocumentData[np.int64]:
    return {
        "doc_id": 1,
        "document_data": [10, 11, 12],
        "document_id": np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32),
        "stream_data": np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64),
    }


@pytest.fixture
def doc_second() -> DocumentData[np.int64]:
    return {
        "doc_id": 2,
        "document_data": [13, 14, 15, 16],
        "document_id": np.array([0, 0, 1, 2, 2, 3, 3, 3], dtype=np.int32),
        "stream_data": np.array([11, 21, 31, 41, 51, 61, 71, 81], dtype=np.int64),
    }


@dataclass
class SimpleExpectation:
    index: int
    length: int
    document_data: list[tuple[int, int]]
    document_id: list[int]
    stream_data: list[int]
    shard: int | None = None

    @property
    def test_name(self) -> str:
        return f"{self.index}-{self.length}"

    def to_dict(self) -> StreamingData[np.int64]:
        return {
            "document_data": [
                DocumentItem(doc_id=doc_id, value=document_value)
                for doc_id, document_value in self.document_data
            ],
            "document_id": np.array(self.document_id, dtype=np.int32),
            "stream_data": np.array(self.stream_data, dtype=np.int64),
        }

    def assert_equal(self, actual: StreamingData[np.int64]) -> None:
        expected = self.to_dict()
        np.testing.assert_array_equal(actual["stream_data"], expected["stream_data"])
        np.testing.assert_array_equal(actual["document_id"], expected["document_id"])
        assert actual["document_data"] == expected["document_data"]
        actual_shard = actual.get("shard", None)
        assert actual_shard is not None
        if self.shard is not None:
            np.testing.assert_array_equal(
                actual_shard, np.full(len(self.document_data), fill_value=self.shard)
            )


first_write_expectations = [
    SimpleExpectation(index=0, length=0, document_data=[], document_id=[], stream_data=[], shard=0),
    SimpleExpectation(index=0, length=1, document_data=[(1, 10)], document_id=[0], stream_data=[10], shard=0),
    SimpleExpectation(
        index=0,
        length=3,
        document_data=[(1, 10), (1, 11)],
        document_id=[0, 0, 1],
        stream_data=[10, 20, 30],
        shard=0,
    ),
    SimpleExpectation(
        index=0,
        length=4,
        document_data=[(1, 10), (1, 11)],
        document_id=[0, 0, 1, 1],
        stream_data=[10, 20, 30, 40],
        shard=0,
    ),
    SimpleExpectation(
        index=0,
        length=7,
        document_data=[(1, 10), (1, 11), (1, 12)],
        document_id=[0, 0, 1, 1, 1, 2, 2],
        stream_data=[10, 20, 30, 40, 50, 60, 70],
        shard=0,
    ),
    SimpleExpectation(
        index=3,
        length=4,
        document_data=[(1, 11), (1, 12)],
        document_id=[0, 0, 1, 1],
        stream_data=[40, 50, 60, 70],
        shard=0,
    ),
]
single_write_expectations = [
    *first_write_expectations,
    SimpleExpectation(
        index=4,
        length=4,
        document_data=[(1, 11), (1, 12), (1, 10)],
        document_id=[0, 1, 1, 2],
        stream_data=[50, 60, 70, 10],
    ),
    SimpleExpectation(
        index=4,
        length=11,
        document_data=[(1, 11), (1, 12), (1, 10), (1, 11), (1, 12), (1, 10)],
        document_id=[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
        stream_data=[50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10],
    ),
    SimpleExpectation(
        index=0,
        length=8,
        document_data=[(1, 10), (1, 11), (1, 12), (1, 10)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 10],
    ),
    SimpleExpectation(
        index=0,
        length=14,
        document_data=[(1, 10), (1, 11), (1, 12), (1, 10), (1, 11), (1, 12)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=0,
        length=15,
        document_data=[(1, 10), (1, 11), (1, 12), (1, 10), (1, 11), (1, 12), (1, 10)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10],
    ),
]


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", single_write_expectations, ids=lambda expected: expected.test_name)
def test_single_write_sync(
    tmp_path: pathlib.Path, expected: SimpleExpectation, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_single_write_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_write_sync", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        assert writer.length(0) == 7

    with storage.sync_reader() as reader:
        result = reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 7
        assert reader.total_length == 7
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", single_write_expectations, ids=lambda expected: expected.test_name)
@pytest.mark.asyncio
async def test_single_write_async(
    tmp_path: pathlib.Path, expected: SimpleExpectation, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_single_write_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_write_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        assert writer.length(0) == 7

    async with storage.async_reader() as reader:
        result = await reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 7
        assert reader.total_length == 7
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_single_write_async_parallel(
    tmp_path: pathlib.Path, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_single_write_async_parallel", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_single_write_async_parallel", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        assert writer.length(0) == 7

    async with storage.async_reader() as reader:
        results = await asyncio.gather(
            *(
                reader.read(global_index=expected.index, global_length=expected.length)
                for expected in single_write_expectations
            )
        )
        for result, expected in zip(results, single_write_expectations, strict=True):
            expected.assert_equal(result)


two_write_expectations = [
    *first_write_expectations,
    SimpleExpectation(
        index=0,
        length=9,
        document_data=[(1, 10), (1, 11), (1, 12), (2, 13)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21],
    ),
    SimpleExpectation(
        index=5,
        length=10,
        document_data=[(1, 12), (2, 13), (2, 14), (2, 15), (2, 16)],
        document_id=[0, 0, 1, 1, 2, 3, 3, 4, 4, 4],
        stream_data=[60, 70, 11, 21, 31, 41, 51, 61, 71, 81],
    ),
    SimpleExpectation(
        index=7,
        length=7,
        document_data=[(2, 13), (2, 14), (2, 15), (2, 16)],
        document_id=[0, 0, 1, 2, 2, 3, 3],
        stream_data=[11, 21, 31, 41, 51, 61, 71],
    ),
    SimpleExpectation(
        index=7,
        length=8,
        document_data=[(2, 13), (2, 14), (2, 15), (2, 16)],
        document_id=[0, 0, 1, 2, 2, 3, 3, 3],
        stream_data=[11, 21, 31, 41, 51, 61, 71, 81],
    ),
    SimpleExpectation(
        index=0,
        length=15,
        document_data=[(1, 10), (1, 11), (1, 12), (2, 13), (2, 14), (2, 15), (2, 16)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81],
    ),
    SimpleExpectation(
        index=0,
        length=16,
        document_data=[(1, 10), (1, 11), (1, 12), (2, 13), (2, 14), (2, 15), (2, 16), (1, 10)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81, 10],
    ),
    SimpleExpectation(
        index=0,
        length=22,
        document_data=[
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (1, 10),
            (1, 11),
            (1, 12),
        ],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=6,
        length=16,
        document_data=[(1, 12), (2, 13), (2, 14), (2, 15), (2, 16), (1, 10), (1, 11), (1, 12)],
        document_id=[0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7],
        stream_data=[70, 11, 21, 31, 41, 51, 61, 71, 81, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=10,
        length=11,
        document_data=[(2, 15), (2, 16), (1, 10), (1, 11), (1, 12)],
        document_id=[0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4],
        stream_data=[41, 51, 61, 71, 81, 10, 20, 30, 40, 50, 60],
    ),
]


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
def test_two_write_one_shard_sync(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_one_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_write_one_shard_sync", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(0, doc_second)
        assert writer.length(0) == 15

    with storage.sync_reader() as reader:
        result = reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 15
        assert reader.total_length == 15
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
@pytest.mark.asyncio
async def test_two_write_one_shard_async(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_one_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_write_one_shard_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(0, doc_second)
        assert writer.length(0) == 15

    async with storage.async_reader() as reader:
        result = await reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 15
        assert reader.total_length == 15
        assert reader.num_shards == 1


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_write_one_shard_async_parallel(
    tmp_path: pathlib.Path,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_one_shard_async_parallel", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_two_write_one_shard_async_parallel", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(0, doc_second)
        assert writer.length(0) == 15

    async with storage.async_reader() as reader:
        results = await asyncio.gather(
            *(
                reader.read(global_index=expected.index, global_length=expected.length)
                for expected in two_write_expectations
            )
        )
        for result, expected in zip(results, two_write_expectations, strict=True):
            expected.assert_equal(result)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
def test_two_write_two_shard_sync(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_write_two_shard_sync", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(1, doc_second)
        assert writer.length(0) == 7
        assert writer.length(1) == 8

    with storage.sync_reader() as reader:
        result = reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert writer.length(0) == 7
        assert writer.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
@pytest.mark.asyncio
async def test_two_write_two_shard_async(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_two_write_two_shard_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(1, doc_second)
        assert writer.length(0) == 7
        assert writer.length(1) == 8

    async with storage.async_reader() as reader:
        result = await reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert writer.length(0) == 7
        assert writer.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_two_write_two_shard_async_parallel(
    tmp_path: pathlib.Path,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_async_parallel", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_two_write_two_shard_async_parallel", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        writer.append(1, doc_second)
        assert writer.length(0) == 7
        assert writer.length(1) == 8

    async with storage.async_reader() as reader:
        results = await asyncio.gather(
            *(
                reader.read(global_index=expected.index, global_length=expected.length)
                for expected in two_write_expectations
            )
        )
        for result, expected in zip(results, two_write_expectations, strict=True):
            expected.assert_equal(result)


#
# Some write specific tests
#


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
def test_two_write_two_shard_seperate_processes_sync(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_seperate_processes_sync", ignore_errors=True)
    with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_sync", stream_dtype=np.dtype(np.int64)
    ).sync_writer() as writer_p0:
        writer_p0.append(0, doc_first)
        assert writer_p0.length(0) == 7

    with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_sync", stream_dtype=np.dtype(np.int64)
    ).sync_writer() as writer_p1:
        writer_p1.append(1, doc_second)
        assert writer_p1.length(1) == 8

    with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_sync", stream_dtype=np.dtype(np.int64)
    ).sync_reader() as reader:
        result = reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 7
        assert reader.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
@pytest.mark.asyncio
async def test_two_write_two_shard_seperate_processes_async(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_seperate_processes_async", ignore_errors=True)
    with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_async", stream_dtype=np.dtype(np.int64)
    ).sync_writer() as writer_p0:
        writer_p0.append(0, doc_first)
        assert writer_p0.length(0) == 7

    with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_async", stream_dtype=np.dtype(np.int64)
    ).sync_writer() as writer_p1:
        writer_p1.append(1, doc_second)
        assert writer_p1.length(1) == 8

    async with io_def.create(
        tmp_path / "test_two_write_two_shard_seperate_processes_async", stream_dtype=np.dtype(np.int64)
    ).async_reader() as reader:
        result = await reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert reader.length(0) == 7
        assert reader.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
def test_two_write_two_shard_reorder_sync(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_reorder_sync", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_two_write_two_shard_reorder_sync", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        writer.append(1, doc_second)
        writer.append(0, doc_first)
        assert writer.length(0) == 7
        assert writer.length(1) == 8

    with storage.sync_reader() as reader:
        result = reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert writer.length(0) == 7
        assert writer.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.parametrize("expected", two_write_expectations, ids=lambda expected: expected.test_name)
@pytest.mark.asyncio
async def test_two_write_two_shard_reorder_async(
    tmp_path: pathlib.Path,
    expected: SimpleExpectation,
    io_def: IoDef,
    doc_first: DocumentData[np.int64],
    doc_second: DocumentData[np.int64],
) -> None:
    shutil.rmtree(tmp_path / "test_two_write_two_shard_reorder_async", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_two_write_two_shard_reorder_async", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        writer.append(1, doc_second)
        writer.append(0, doc_first)
        assert writer.length(0) == 7
        assert writer.length(1) == 8

    async with storage.async_reader() as reader:
        result = await reader.read(global_index=expected.index, global_length=expected.length)
        expected.assert_equal(result)
        assert writer.length(0) == 7
        assert writer.length(1) == 8
        assert reader.total_length == 15
        assert reader.num_shards == 2


#
# Edge cases
#
@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_empty_shard_sync(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_empty_shard_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_empty_shard_sync", stream_dtype=np.dtype(np.int64))

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
    storage = io_def.create(tmp_path / "test_empty_shard_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as reader:
        pass

    async with storage.async_reader() as reader:
        assert reader.num_shards == 0
        assert reader.total_length == 0

        with pytest.raises(IndexError, match="out of range"):
            reader.length(0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_zero_length_data_sync(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_zero_length_data_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_zero_length_data_sync", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(
            0,
            {
                "doc_id": 99,
                "document_data": [],
                "document_id": np.array([], dtype=np.int32),
                "stream_data": np.array([], dtype=np.int64),
            },
        )

    with storage.sync_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=0, global_length=0)

        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=0, global_length=1)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_zero_length_data_async(tmp_path: pathlib.Path, io_def: IoDef) -> None:
    shutil.rmtree(tmp_path / "test_zero_length_data_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_zero_length_data_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(
            0,
            {
                "doc_id": 99,
                "document_data": [],
                "document_id": np.array([], dtype=np.int32),
                "stream_data": np.array([], dtype=np.int64),
            },
        )

    async with storage.async_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=0, global_length=0)

        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=0, global_length=1)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_out_of_bounds_read_sync(
    tmp_path: pathlib.Path, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_read_sync", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_read_sync", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        assert writer.length(0) == 7

    # Invalid read
    with storage.sync_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=7, global_length=1)

        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=10, global_length=1)

        with pytest.raises(IndexError, match="outside of size"):
            reader.read(global_index=10, global_length=0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_out_of_bounds_read_async(
    tmp_path: pathlib.Path, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_out_of_bounds_read_async", ignore_errors=True)
    storage = io_def.create(tmp_path / "test_out_of_bounds_read_async", stream_dtype=np.dtype(np.int64))

    with storage.sync_writer() as writer:
        writer.append(0, doc_first)
        assert writer.length(0) == 7

    # Invalid read
    async with storage.async_reader() as reader:
        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=7, global_length=1)

        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=10, global_length=1)

        with pytest.raises(IndexError, match="outside of size"):
            await reader.read(global_index=10, global_length=0)


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
def test_non_sequentally_unique_keys_sync(
    tmp_path: pathlib.Path, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_non_sequentally_unique_keys_sync", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_non_sequentally_unique_keys_sync", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        with pytest.raises(ValueError, match="keys must be sequentually unique"):
            writer.append(
                0,
                {
                    "doc_id": 1,
                    "document_data": [10, 11, 12],
                    "document_id": np.array([0, 0, 1, 1, 0, 2, 2], dtype=np.int32),
                    "stream_data": np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64),
                },
            )


@pytest.mark.parametrize("io_def", io_interfaces, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_non_sequentally_unique_keys_async(
    tmp_path: pathlib.Path, io_def: IoDef, doc_first: DocumentData[np.int64]
) -> None:
    shutil.rmtree(tmp_path / "test_non_sequentally_unique_keys_async", ignore_errors=True)
    storage = io_def.create(
        tmp_path / "test_non_sequentally_unique_keys_async", stream_dtype=np.dtype(np.int64)
    )

    with storage.sync_writer() as writer:
        with pytest.raises(ValueError, match="keys must be sequentually unique"):
            writer.append(
                0,
                {
                    "doc_id": 1,
                    "document_data": [10, 11, 12],
                    "document_id": np.array([0, 0, 1, 1, 0, 2, 2], dtype=np.int32),
                    "stream_data": np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64),
                },
            )


def test_pack_yields_wrong_number_of_pairs_sync(tmp_path: pathlib.Path) -> None:
    # The test format sizes a record by `stream_data` and yields one pair per `document_id` entry.
    storage = StreamWithMetaStorage(
        tmp_path / "fewer_pairs", ConceptStorageFormat(stream_dtype=np.dtype(np.int64))
    )

    with storage.sync_writer() as writer:
        with pytest.raises(ValueError, match="yielded 2"):
            writer.append(
                0,
                {
                    "doc_id": 1,
                    "document_data": [10],
                    "document_id": np.array([0, 0], dtype=np.int32),
                    "stream_data": np.arange(4, dtype=np.int64),
                },
            )

        with pytest.raises(ValueError, match="yielded more"):
            writer.append(
                0,
                {
                    "doc_id": 1,
                    "document_data": [10],
                    "document_id": np.array([0, 0, 0, 0], dtype=np.int32),
                    "stream_data": np.arange(2, dtype=np.int64),
                },
            )


class ReturnTransformFormat(ConceptStorageFormat[np.int64]):
    """Yields the same pairs as `ConceptStorageFormat`, then transforms the returned stream array."""

    def __init__(self, transform: Callable[[Vector[np.int64]], object]) -> None:
        super().__init__(stream_dtype=np.dtype(np.int64))
        self.transform = transform

    def pack(
        self, data: DocumentData[np.int64]
    ) -> Generator[tuple[Hashable, UserBufferTypes | None], None, Vector[np.int64]]:
        stream_data = yield from super().pack(data)
        return self.transform(stream_data)  # pyright: ignore[reportReturnType] -- intentional error


@pytest.mark.parametrize(
    ("transform", "error", "match"),
    [
        pytest.param(lambda stream: None, TypeError, "got NoneType", id="missing-return"),
        pytest.param(lambda stream: stream[:1], ValueError, "got shape", id="wrong-shape"),
        pytest.param(
            lambda stream: stream.astype(np.int32), ValueError, "returned dtype int32", id="wrong-dtype"
        ),
    ],
)
def test_pack_returns_invalid_stream_sync(
    tmp_path: pathlib.Path,
    doc_first: DocumentData[np.int64],
    transform: Callable[[Vector[np.int64]], object],
    error: type[Exception],
    match: str,
) -> None:
    # Previously, a missing `return` failed as "not fully consumed", a length-1 array was broadcast
    # to every stream element, and another dtype was silently cast.
    storage = StreamWithMetaStorage(tmp_path / "invalid", ReturnTransformFormat(transform))

    with storage.sync_writer() as writer:
        with pytest.raises(error, match=match):
            writer.append(0, doc_first)
