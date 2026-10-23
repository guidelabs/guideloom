from dataclasses import dataclass
import pathlib

import numpy as np
import pytest

from guideloom.filesystem import FileSystemStream
from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.types import IOReaderAbstract


@dataclass
class IOInterface:
    interface: IOReaderAbstract | None
    name: str


metal_interface: list[IOInterface] = [
    IOInterface(None, "default"),
    IOInterface(MmapIO(), "mmap"),
    IOInterface(SeekIO(), "seek"),
]
if PosixPreadIO.is_supported(advice=False):
    metal_interface.append(IOInterface(PosixPreadIO(advice=False), "pread"))
if PosixPreadIO.is_supported(advice=True):
    metal_interface.append(IOInterface(PosixPreadIO(advice=True), "advice"))


# async tests
@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_read_structured_array(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "data.bin"

    dtype = np.dtype([("x", np.int32), ("y", np.float64)])
    data = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)

    with FileSystemStream(path, read_interface=metal.interface).sync_writer(dtype) as writer:
        writer.append(0, data)

    with FileSystemStream(path, read_interface=metal.interface).sync_reader(dtype) as reader:
        out = reader.read(0, 0, 3)
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_read_structured_array(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "data.bin"

    dtype = np.dtype([("x", np.int32), ("y", np.float64)])
    data = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)

    with FileSystemStream(path, read_interface=metal.interface).sync_writer(dtype) as writer:
        writer.append(0, data)

    async with FileSystemStream(path, read_interface=metal.interface).async_reader(dtype) as reader:
        out = await reader.read(0, 0, 3)
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_distinct_shards(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "shards.bin"

    dtype = np.dtype([("x", np.int32), ("y", np.float64)])
    data_0 = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)
    data_1 = np.array([(4, 4.5), (5, 5.5), (6, 6.5)], dtype=dtype)

    with FileSystemStream(path, read_interface=metal.interface).sync_writer(dtype) as writer:
        writer.append(0, data_0)
        writer.append(1, data_1)

    with FileSystemStream(path, read_interface=metal.interface).sync_reader(dtype) as reader:
        out_0 = reader.read(0, 0, 3)
        out_1 = reader.read(1, 0, 3)
    np.testing.assert_array_equal(out_0, data_0)
    np.testing.assert_array_equal(out_1, data_1)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_distinct_shards(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "shards.bin"

    dtype = np.dtype([("x", np.int32), ("y", np.float64)])
    data_0 = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)
    data_1 = np.array([(4, 4.5), (5, 5.5), (6, 6.5)], dtype=dtype)

    with FileSystemStream(path, read_interface=metal.interface).sync_writer(dtype) as writer:
        writer.append(0, data_0)
        writer.append(1, data_1)

    async with FileSystemStream(path, read_interface=metal.interface).async_reader(dtype) as reader:
        out_0 = await reader.read(0, 0, 3)
        out_1 = await reader.read(1, 0, 3)

    np.testing.assert_array_equal(out_0, data_0)
    np.testing.assert_array_equal(out_1, data_1)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_byte_alignment(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "offset.bin"
    dtype = np.dtype([("x", np.int32), ("y", np.int64)])

    fs = FileSystemStream(path, read_interface=metal.interface)
    arr = np.array([(i * 5 + 7, i * 11 + 13) for i in range(15)], dtype=dtype)

    with fs.sync_writer(dtype) as writer:
        writer.append(0, arr[:10])
        writer.append(0, arr[10:])

    with fs.sync_reader(dtype) as reader:
        for start, end in [(0, 15), (0, 5), (3, 5), (7, 10), (7, 12), (10, 15)]:
            middle = reader.read(0, start, end - start)
            expected = arr[start:end]
            np.testing.assert_array_equal(middle, expected)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_byte_alignment(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "offset.bin"
    dtype = np.dtype([("x", np.int32), ("y", np.int64)])

    fs = FileSystemStream(path, read_interface=metal.interface)
    arr = np.array([(i * 5 + 7, i * 11 + 13) for i in range(15)], dtype=dtype)

    with fs.sync_writer(dtype) as writer:
        writer.append(0, arr[:10])
        writer.append(0, arr[10:])

    async with fs.async_reader(dtype) as reader:
        for start, end in [(0, 15), (0, 5), (3, 5), (7, 10), (7, 12), (10, 15)]:
            middle = await reader.read(0, start, end - start)
            expected = arr[start:end]
            np.testing.assert_array_equal(middle, expected)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_read_chunk_out_of_range(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "out_of_range.bin"
    dtype = np.dtype([("value", np.int32)])

    fs = FileSystemStream(path, read_interface=metal.interface)
    data = np.array([(1,), (2,), (3,)], dtype=dtype)

    # Write only 3 elements
    with fs.sync_writer(dtype) as writer:
        writer.append(0, data)

    with fs.sync_reader(dtype) as reader:
        with pytest.raises(IndexError, match="shorter than requested size"):
            reader.read(0, 10, 5)

        with pytest.raises(IndexError, match="Cannot read negative index"):
            reader.read(0, -10, 5)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_read_chunk_out_of_range(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "out_of_range.bin"
    dtype = np.dtype([("value", np.int32)])

    fs = FileSystemStream(path, read_interface=metal.interface)
    data = np.array([(1,), (2,), (3,)], dtype=dtype)

    # Write only 3 elements
    with fs.sync_writer(dtype) as writer:
        writer.append(0, data)

    async with fs.async_reader(dtype) as reader:
        with pytest.raises(IndexError, match="shorter than requested size"):
            await reader.read(0, 10, 5)

        with pytest.raises(IndexError, match="Cannot read negative index"):
            await reader.read(0, -10, 5)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_write_cast_dtype(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "data_wrong_dtype.bin"
    dtype_le = np.dtype([("x", np.dtype("<i4"))])
    data_le = np.array([(1,), (2,)], dtype=dtype_le)
    dtype_ge = np.dtype([("x", np.dtype(">i4"))])
    data_ge = np.array([(1,), (2,)], dtype=dtype_ge)

    fs_le = FileSystemStream(path, read_interface=metal.interface)
    with fs_le.sync_writer(dtype_le) as writer:
        writer.append(0, data_ge)

    with fs_le.sync_reader(dtype_le) as reader_le:
        content_le = reader_le.read(0, 0, 2)
        np.testing.assert_array_equal(content_le, data_le)

    fs_ge = FileSystemStream(path, read_interface=metal.interface)
    with fs_ge.sync_reader(dtype_ge) as reader_ge:
        content_ge = reader_ge.read(0, 0, 2)
        np.testing.assert_array_equal(content_ge, data_le)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_write_cast_dtype(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    path = tmp_path / "data_wrong_dtype.bin"
    dtype_le = np.dtype([("x", np.dtype("<i4"))])
    data_le = np.array([(1,), (2,)], dtype=dtype_le)
    dtype_ge = np.dtype([("x", np.dtype(">i4"))])
    data_ge = np.array([(1,), (2,)], dtype=dtype_ge)

    fs_le = FileSystemStream(path, read_interface=metal.interface)
    with fs_le.sync_writer(dtype_le) as writer:
        writer.append(0, data_ge)

    async with fs_le.async_reader(dtype_le) as reader_le:
        content_le = await reader_le.read(0, 0, 2)
        np.testing.assert_array_equal(content_le, data_le)

    fs_ge = FileSystemStream(path, read_interface=metal.interface)
    async with fs_ge.async_reader(dtype_ge) as reader_ge:
        content_ge = await reader_ge.read(0, 0, 2)
        np.testing.assert_array_equal(content_ge, data_le)


# error tests
def test_write_wrong_dtype_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data_wrong_dtype.bin"
    dtype = np.dtype([("x", np.int32)])

    with FileSystemStream(path).sync_writer(dtype) as writer:
        with pytest.raises(ValueError, match="unexpected dtype"):
            wrong_dtype = np.dtype([("x", np.int64)])
            data = np.array([(1,), (2,)], dtype=wrong_dtype)
            writer.append(0, data)


def test_write_wrong_dim_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data_wrong_dim.bin"
    dtype = np.dtype([("x", np.int32)])

    bad_data = np.array([[(1,), (2,)]], dtype=dtype)

    with FileSystemStream(path).sync_writer(dtype) as writer:
        with pytest.raises(ValueError, match="must be 1-D"):
            writer.append(0, bad_data)
