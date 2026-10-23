from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
import pathlib

import numpy as np
import pytest

from guideloom.filesystem import FileSystemDocument
from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.types import IOReaderAbstract, StorageBufferTypes, Vector


@dataclass
class DBIODef:
    name: str
    constructor: Callable[..., FileSystemDocument]
    read_interface: IOReaderAbstract | None = None

    def Create(self, path: pathlib.Path) -> FileSystemDocument:
        if self.read_interface is None:
            return self.constructor(path)
        return self.constructor(path, read_interface=self.read_interface)


def _pack_requests(requests: Sequence[tuple[int, int]]) -> Vector[np.void]:
    pack = np.empty(
        len(requests),
        dtype=np.dtype(
            [
                ("shard", np.int64),
                ("shard_index", np.int32),
            ],
            align=True,
        ),
    )
    pack["shard"] = [shard for shard, _ in requests]
    pack["shard_index"] = [index for _, index in requests]
    return pack


def _assert_memoryview_equal(actual: Sequence[StorageBufferTypes], expected: Sequence[bytes]) -> None:
    actual_convereted = [bytes(item) for item in actual]
    assert actual_convereted == expected


async_document: list[DBIODef] = [
    DBIODef("fs-default", FileSystemDocument),
    DBIODef("fs-mmap", FileSystemDocument, MmapIO()),
    DBIODef("fs-seek", FileSystemDocument, SeekIO()),
]

if PosixPreadIO.is_supported(advice=False):
    async_document.append(DBIODef("fs-pread", FileSystemDocument, PosixPreadIO(advice=False)))
if PosixPreadIO.is_supported(advice=True):
    async_document.append(DBIODef("fs-advice", FileSystemDocument, PosixPreadIO(advice=True)))

sync_document: list[DBIODef] = [*async_document]


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_write_and_read_scalar(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_write_and_read_scalar")
    value = b"CUSTOM VALUE"

    with db.sync_writer() as writer:
        writer.append(0, value)
        assert writer.length(0) == 1

    with db.sync_reader() as reader:
        assert reader.length(0) == 1
        assert reader.num_shards() == 1

        assert reader.get(0, 0) == value
        _assert_memoryview_equal(reader.get_many(_pack_requests([(0, 0)])), [value])


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_write_and_read_scalar(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_write_and_read_scalar")
    value = b"CUSTOM VALUE"

    with db.sync_writer() as writer:
        writer.append(0, value)
        assert writer.length(0) == 1

    async with db.async_reader() as reader:
        assert await reader.length(0) == 1
        assert await reader.num_shards() == 1

        assert await reader.get(0, 0) == value
        _assert_memoryview_equal(await reader.get_many(_pack_requests([(0, 0)])), [value])


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_multi_write_and_read(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_multi_write_and_read")
    value_0 = b"VALUE ZERO"
    value_1 = b"VALUE ONE"
    value_2 = b"VALUE TWO"
    value_3 = b"VALUE THREE"

    with db.sync_writer() as writer:
        writer.append(0, value_0)
        writer.append(0, value_1)
        writer.append(0, value_2)
        writer.append(0, value_3)
        assert writer.length(0) == 4

    with db.sync_reader() as reader:
        assert reader.length(0) == 4
        assert reader.num_shards() == 1

        assert reader.get(0, 0) == value_0
        assert reader.get(0, 1) == value_1
        assert reader.get(0, 2) == value_2
        assert reader.get(0, 3) == value_3
        _assert_memoryview_equal(
            reader.get_many(_pack_requests([(0, 0), (0, 1), (0, 2), (0, 3)])),
            [value_0, value_1, value_2, value_3],
        )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_multi_write_and_read(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_multi_write_and_read")
    value_0 = b"VALUE ZERO"
    value_1 = b"VALUE ONE"
    value_2 = b"VALUE TWO"
    value_3 = b"VALUE THREE"

    with db.sync_writer() as writer:
        writer.append(0, value_0)
        writer.append(0, value_1)
        writer.append(0, value_2)
        writer.append(0, value_3)
        assert writer.length(0) == 4

    async with db.async_reader() as reader:
        assert await reader.length(0) == 4
        assert await reader.num_shards() == 1

        assert await reader.get(0, 0) == value_0
        assert await reader.get(0, 1) == value_1
        assert await reader.get(0, 2) == value_2
        assert await reader.get(0, 3) == value_3
        _assert_memoryview_equal(
            await reader.get_many(_pack_requests([(0, 0), (0, 1), (0, 2), (0, 3)])),
            [value_0, value_1, value_2, value_3],
        )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_sync_multi_write_and_read_permutate_read(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_multi_write_and_read_permutate_read")
    value_0 = b"VALUE ZERO"
    value_1 = b"VALUE ONE"
    value_2 = b"VALUE TWO"
    value_3 = b"VALUE THREE"

    with db.sync_writer() as writer:
        writer.append(0, value_0)
        writer.append(0, value_1)
        writer.append(0, value_2)
        writer.append(0, value_3)
        assert writer.length(0) == 4

    tests = [(0, 0, value_0), (0, 1, value_1), (0, 2, value_2), (0, 3, value_3)]
    with db.sync_reader() as reader:
        assert reader.length(0) == 4
        assert reader.num_shards() == 1

        for tests_permuted in permutations(tests):
            for shard, index, expected in tests_permuted:
                assert reader.get(shard, index) == expected
            _assert_memoryview_equal(
                reader.get_many(_pack_requests([(shard, index) for shard, index, _ in tests_permuted])),
                [value for _, _, value in tests_permuted],
            )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_multi_write_and_read_permutate_read(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_multi_write_and_read_permutate_read")
    value_0 = b"VALUE ZERO"
    value_1 = b"VALUE ONE"
    value_2 = b"VALUE TWO"
    value_3 = b"VALUE THREE"

    with db.sync_writer() as writer:
        writer.append(0, value_0)
        writer.append(0, value_1)
        writer.append(0, value_2)
        writer.append(0, value_3)
        assert writer.length(0) == 4

    tests = [(0, 0, value_0), (0, 1, value_1), (0, 2, value_2), (0, 3, value_3)]
    async with db.async_reader() as reader:
        assert await reader.length(0) == 4
        assert await reader.num_shards() == 1

        for tests_permuted in permutations(tests):
            for shard, index, expected in tests_permuted:
                assert await reader.get(shard, index) == expected
            _assert_memoryview_equal(
                await reader.get_many(_pack_requests([(shard, index) for shard, index, _ in tests_permuted])),
                [value for _, _, value in tests_permuted],
            )


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_write_and_read_multiple_shards(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_write_and_read_multiple_shards")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(0, value_01)
        writer.append(1, value_10)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    with db.sync_reader() as reader:
        assert reader.length(0) == 2
        assert reader.length(1) == 2
        assert reader.num_shards() == 2

        assert reader.get(0, 0) == value_00
        assert reader.get(0, 1) == value_01
        assert reader.get(1, 0) == value_10
        assert reader.get(1, 1) == value_11
        _assert_memoryview_equal(
            reader.get_many(_pack_requests([(0, 0), (0, 1), (1, 0), (1, 1)])),
            [value_00, value_01, value_10, value_11],
        )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_write_and_read_multiple_shards(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_write_and_read_multiple_shards")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(0, value_01)
        writer.append(1, value_10)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    async with db.async_reader() as reader:
        assert await reader.length(0) == 2
        assert await reader.length(1) == 2
        assert await reader.num_shards() == 2

        assert await reader.get(0, 0) == value_00
        assert await reader.get(0, 1) == value_01
        assert await reader.get(1, 0) == value_10
        assert await reader.get(1, 1) == value_11
        _assert_memoryview_equal(
            await reader.get_many(_pack_requests([(0, 0), (0, 1), (1, 0), (1, 1)])),
            [value_00, value_01, value_10, value_11],
        )


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_write_and_read_multiple_shards_reorder(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_write_and_read_multiple_shards_reorder")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(1, value_10)
        writer.append(0, value_01)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    with db.sync_reader() as reader:
        assert reader.length(0) == 2
        assert reader.length(1) == 2
        assert reader.num_shards() == 2

        assert reader.get(0, 0) == value_00
        assert reader.get(0, 1) == value_01
        assert reader.get(1, 0) == value_10
        assert reader.get(1, 1) == value_11
        _assert_memoryview_equal(
            reader.get_many(_pack_requests([(0, 0), (0, 1), (1, 0), (1, 1)])),
            [value_00, value_01, value_10, value_11],
        )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_write_and_read_multiple_shards_reorder(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_write_and_read_multiple_shards_reorder")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(1, value_10)
        writer.append(0, value_01)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    async with db.async_reader() as reader:
        assert await reader.length(0) == 2
        assert await reader.length(1) == 2
        assert await reader.num_shards() == 2

        assert await reader.get(0, 0) == value_00
        assert await reader.get(0, 1) == value_01
        assert await reader.get(1, 0) == value_10
        assert await reader.get(1, 1) == value_11
        _assert_memoryview_equal(
            await reader.get_many(_pack_requests([(0, 0), (0, 1), (1, 0), (1, 1)])),
            [value_00, value_01, value_10, value_11],
        )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_sync_write_and_read_multiple_shards_permuate_read(
    tmp_path: pathlib.Path, dbio: DBIODef
) -> None:
    db = dbio.Create(tmp_path / "test_sync_write_and_read_multiple_shards_permuate_read")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(1, value_10)
        writer.append(0, value_01)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    tests = [(0, 0, value_00), (0, 1, value_01), (1, 0, value_10), (1, 1, value_11)]
    with db.sync_reader() as reader:
        assert reader.length(0) == 2
        assert reader.length(1) == 2
        assert reader.num_shards() == 2

        for tests_permuted in permutations(tests):
            for shard, index, expected in tests_permuted:
                assert reader.get(shard, index) == expected
            _assert_memoryview_equal(
                reader.get_many(_pack_requests([(shard, index) for shard, index, _ in tests_permuted])),
                [value for _, _, value in tests_permuted],
            )


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_write_and_read_multiple_shards_permuate_read(
    tmp_path: pathlib.Path, dbio: DBIODef
) -> None:
    db = dbio.Create(tmp_path / "test_async_write_and_read_multiple_shards_permuate_read")
    value_00 = b"VALUE ZERO ZERO"
    value_01 = b"VALUE ZERO ONE"
    value_10 = b"VALUE ONE ZERO"
    value_11 = b"VALUE ONE ONE"

    with db.sync_writer() as writer:
        writer.append(0, value_00)
        writer.append(1, value_10)
        writer.append(0, value_01)
        writer.append(1, value_11)
        assert writer.length(0) == 2
        assert writer.length(1) == 2

    tests = [(0, 0, value_00), (0, 1, value_01), (1, 0, value_10), (1, 1, value_11)]
    async with db.async_reader() as reader:
        assert await reader.length(0) == 2
        assert await reader.length(1) == 2
        assert await reader.num_shards() == 2

        for tests_permuted in permutations(tests):
            for shard, index, expected in tests_permuted:
                assert await reader.get(shard, index) == expected
            _assert_memoryview_equal(
                await reader.get_many(_pack_requests([(shard, index) for shard, index, _ in tests_permuted])),
                [value for _, _, value in tests_permuted],
            )


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_get_invalid_key_after_raises(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_get_invalid_key_after_raises")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")
        assert writer.length(0) == 1

    with db.sync_reader() as reader:
        assert reader.length(0) == 1

        with pytest.raises(IndexError):
            reader.get(0, 1)

        with pytest.raises(IndexError):
            reader.get_many(_pack_requests([(0, 1)]))


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_get_invalid_key_after_raises(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_get_invalid_key_after_raises")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")
        assert writer.length(0) == 1

    async with db.async_reader() as reader:
        assert await reader.length(0) == 1

        with pytest.raises(IndexError):
            await reader.get(0, 1)

        with pytest.raises(IndexError):
            await reader.get_many(_pack_requests([(0, 1)]))


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
def test_sync_empty_read_many(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_empty_read_many")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")

    with db.sync_reader() as reader:
        assert reader.get_many(_pack_requests([])) == []


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_empty_read_many(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_empty_read_many")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")

    async with db.async_reader() as reader:
        assert await reader.get_many(_pack_requests([])) == []


@pytest.mark.parametrize("dbio", sync_document, ids=lambda dbio: dbio.name)
def test_sync_get_invalid_shard_length(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_sync_get_invalid_shard_length")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")
        assert writer.length(0) == 1

    with db.sync_reader() as reader:
        assert reader.length(0) == 1
        with pytest.raises(IndexError):
            reader.length(1)


@pytest.mark.parametrize("dbio", async_document, ids=lambda dbio: dbio.name)
@pytest.mark.asyncio
async def test_async_get_invalid_shard_length(tmp_path: pathlib.Path, dbio: DBIODef) -> None:
    db = dbio.Create(tmp_path / "test_async_get_invalid_shard_length")

    with db.sync_writer() as writer:
        writer.append(0, b"VALUE")
        assert writer.length(0) == 1

    async with db.async_reader() as reader:
        assert await reader.length(0) == 1
        with pytest.raises(IndexError):
            await reader.length(1)
