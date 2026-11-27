from collections import Counter
import asyncio
import pathlib

import numpy as np
import pytest
import aiofiles.os

from guideloom.filesystem import FileSystemDocument, FileSystemStream


@pytest.fixture
def open_counts(monkeypatch: pytest.MonkeyPatch) -> Counter[pathlib.Path]:
    counts: Counter[pathlib.Path] = Counter()
    exists = aiofiles.os.path.exists

    async def record_open(path: pathlib.Path) -> bool:
        counts[path] += 1
        # Give competing reads a chance to reach their first shard open.
        await asyncio.sleep(0)
        return bool(await exists(path))

    monkeypatch.setattr(aiofiles.os.path, "exists", record_open)
    return counts


@pytest.mark.asyncio
async def test_concurrent_stream_reads_open_each_shard_once(
    tmp_path: pathlib.Path, open_counts: Counter[pathlib.Path]
) -> None:
    dtype = np.dtype(np.int32)
    storage = FileSystemStream(tmp_path / "stream")
    with storage.sync_writer(dtype) as writer:
        writer.append(0, np.array([10, 20], dtype=dtype))
        writer.append(1, np.array([30], dtype=dtype))

    async with storage.async_reader(dtype) as reader:
        first, second, other_shard = await asyncio.gather(
            reader.read(0, 0, 1), reader.read(0, 1, 1), reader.read(1, 0, 1)
        )
        assert [first.item(), second.item(), other_shard.item()] == [10, 20, 30]

    assert open_counts == {storage.path / f"{shard:07d}.in_band.bin": 1 for shard in (0, 1)}


@pytest.mark.asyncio
async def test_nonconsecutive_document_reads_open_each_file_once(
    tmp_path: pathlib.Path, open_counts: Counter[pathlib.Path]
) -> None:
    storage = FileSystemDocument(tmp_path / "documents")
    with storage.sync_writer() as writer:
        for value in (b"a", b"b", b"c"):
            writer.append(0, value)
        writer.append(1, b"d")

    requests = np.array(
        [(0, 2), (1, 0), (0, 0), (0, 2)],
        dtype=np.dtype([("shard", np.int64), ("shard_index", np.int32)], align=True),
    )
    async with storage.async_reader() as reader:
        batch, single = await asyncio.gather(reader.get_many(requests), reader.get(0, 1))
        assert [bytes(value) for value in batch] == [b"c", b"d", b"a", b"c"]
        assert bytes(single) == b"b"

    assert open_counts == {
        storage.path / f"{shard:07d}.out_of_band.{suffix}": 1
        for shard in (0, 1)
        for suffix in ("index", "bin")
    }


def test_stream_reader_reopens_on_a_new_event_loop(
    tmp_path: pathlib.Path, open_counts: Counter[pathlib.Path]
) -> None:
    dtype = np.dtype(np.int32)
    storage = FileSystemStream(tmp_path / "stream")
    with storage.sync_writer(dtype) as writer:
        writer.append(0, np.array([10, 20], dtype=dtype))
    reader = storage.async_reader(dtype)

    async def read_shard() -> list[int]:
        async with reader:
            first, second = await asyncio.gather(reader.read(0, 0, 1), reader.read(0, 1, 1))
            return [first.item(), second.item()]

    assert asyncio.run(read_shard()) == [10, 20]
    assert asyncio.run(read_shard()) == [10, 20]
