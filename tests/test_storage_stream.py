from dataclasses import dataclass
from typing import NotRequired, TypedDict
import shutil
import asyncio
import pathlib

import numpy as np
import pytest

from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.storage import StreamStorage
from guideloom.types import IOReaderAbstract, StreamStorageFormat, Vector


#
# The ConceptStorageFormat applied thoughout this test, fairly basic. But should cover everything.
#
class DocumentData[InputDType: np.generic](TypedDict):
    doc_id: int
    stream_data: Vector[InputDType]


class StreamingData[InputDType: np.generic](TypedDict):
    shard_ids: NotRequired[Vector[np.int32]]
    doc_ids: Vector[np.int32]
    stream_data: Vector[InputDType]


class ConceptStorageFormat[DataDType: np.generic](
    StreamStorageFormat[DocumentData[DataDType], StreamingData[DataDType], np.void]
):
    def __init__(self, stream_dtype: np.dtype[DataDType]) -> None:
        structured_dtype = np.dtype([("doc_id", np.int32), ("stream_data", stream_dtype)])
        super().__init__(structured_dtype)

    def pack(self, data: DocumentData[DataDType]) -> Vector[np.void]:
        out = np.empty((data["stream_data"].size,), dtype=self.stream_dtype)
        out["stream_data"] = data["stream_data"]
        out["doc_id"] = data["doc_id"]
        return out

    def unpack(
        self, stream: Vector[np.void], shard_lengths: Vector[np.int64], shards: Vector[np.int64]
    ) -> StreamingData[DataDType]:
        shards_np: Vector[np.int32]
        if len(shards) == 0:
            shards_np = np.zeros(shape=(0,), dtype=np.int32)
        else:
            shards_np = np.concat(
                [
                    np.full(shape=(length,), fill_value=shard, dtype=np.int32)
                    for length, shard in zip(shard_lengths, shards, strict=True)
                ],
                axis=0,
            )

        return {"shard_ids": shards_np, "doc_ids": stream["doc_id"], "stream_data": stream["stream_data"]}


#
# I/O interface
#
@dataclass
class IoDef:
    name: str
    fs_read_interface: IOReaderAbstract[pathlib.Path] | None = None

    def create[DType: np.generic](
        self, path: pathlib.Path, stream_dtype: np.dtype[DType]
    ) -> StreamStorage[DocumentData[DType], StreamingData[DType], np.void]:
        formatter = ConceptStorageFormat(stream_dtype=stream_dtype)
        return StreamStorage(path, formatter, read_interface=self.fs_read_interface)


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
    return {"doc_id": 1, "stream_data": np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)}


@pytest.fixture
def doc_second() -> DocumentData[np.int64]:
    return {"doc_id": 2, "stream_data": np.array([11, 21, 31, 41, 51, 61, 71, 81], dtype=np.int64)}


@dataclass
class SimpleExpectation:
    index: int
    length: int
    doc_id: list[int]
    stream_data: list[int]
    shard: int | None = None

    @property
    def test_name(self) -> str:
        return f"{self.index}-{self.length}"

    def to_dict(self) -> StreamingData[np.int64]:
        return {
            "doc_ids": np.array(self.doc_id, dtype=np.int32),
            "stream_data": np.array(self.stream_data, dtype=np.int64),
        }

    def assert_equal(self, actual: StreamingData[np.int64]) -> None:
        expected = self.to_dict()
        np.testing.assert_array_equal(actual["doc_ids"], expected["doc_ids"])
        np.testing.assert_array_equal(actual["stream_data"], expected["stream_data"])

        actual_shard = actual.get("shard_ids", None)
        assert actual_shard is not None
        assert actual_shard.size == expected["stream_data"].size

        if self.shard is not None:
            np.testing.assert_array_equal(
                actual_shard, np.full((len(self.stream_data),), fill_value=self.shard, dtype=np.int32)
            )


first_write_expectations = [
    SimpleExpectation(index=0, length=0, doc_id=[], stream_data=[], shard=0),
    SimpleExpectation(index=0, length=1, doc_id=[1], stream_data=[10], shard=0),
    SimpleExpectation(index=0, length=3, doc_id=[1, 1, 1], stream_data=[10, 20, 30], shard=0),
    SimpleExpectation(index=0, length=4, doc_id=[1, 1, 1, 1], stream_data=[10, 20, 30, 40], shard=0),
    SimpleExpectation(
        index=0, length=7, doc_id=[1, 1, 1, 1, 1, 1, 1], stream_data=[10, 20, 30, 40, 50, 60, 70], shard=0
    ),
    SimpleExpectation(index=3, length=4, doc_id=[1, 1, 1, 1], stream_data=[40, 50, 60, 70], shard=0),
]
single_write_expectations = [
    *first_write_expectations,
    SimpleExpectation(index=4, length=4, doc_id=[1, 1, 1, 1], stream_data=[50, 60, 70, 10]),
    SimpleExpectation(
        index=4,
        length=11,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        stream_data=[50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10],
    ),
    SimpleExpectation(
        index=0, length=8, doc_id=[1, 1, 1, 1, 1, 1, 1, 1], stream_data=[10, 20, 30, 40, 50, 60, 70, 10]
    ),
    SimpleExpectation(
        index=0,
        length=14,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=0,
        length=15,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
        doc_id=[1, 1, 1, 1, 1, 1, 1, 2, 2],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21],
    ),
    SimpleExpectation(
        index=5,
        length=10,
        doc_id=[1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        stream_data=[60, 70, 11, 21, 31, 41, 51, 61, 71, 81],
    ),
    SimpleExpectation(
        index=7, length=7, doc_id=[2, 2, 2, 2, 2, 2, 2], stream_data=[11, 21, 31, 41, 51, 61, 71]
    ),
    SimpleExpectation(
        index=7, length=8, doc_id=[2, 2, 2, 2, 2, 2, 2, 2], stream_data=[11, 21, 31, 41, 51, 61, 71, 81]
    ),
    SimpleExpectation(
        index=0,
        length=15,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81],
    ),
    SimpleExpectation(
        index=0,
        length=16,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81, 10],
    ),
    SimpleExpectation(
        index=0,
        length=22,
        doc_id=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        stream_data=[10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 81, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=6,
        length=16,
        doc_id=[1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        stream_data=[70, 11, 21, 31, 41, 51, 61, 71, 81, 10, 20, 30, 40, 50, 60, 70],
    ),
    SimpleExpectation(
        index=10,
        length=11,
        doc_id=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
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
        writer.append(0, {"doc_id": 99, "stream_data": np.array([], dtype=np.int64)})

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
        writer.append(0, {"doc_id": 99, "stream_data": np.array([], dtype=np.int64)})

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
