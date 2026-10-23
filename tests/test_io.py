from collections.abc import Generator
from dataclasses import dataclass
from types import TracebackType
import pathlib

import pytest

from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.types import IOAsyncReaderAbstract, IOReaderAbstract, IOSyncReaderAbstract


class AutoSyncOpen:
    def __init__(self, obj: IOSyncReaderAbstract) -> None:
        self.obj = obj

    def __enter__(self) -> IOSyncReaderAbstract:
        self.obj.open()
        return self.obj

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.obj.close()


class AutoAsyncOpen:
    def __init__(self, obj: IOAsyncReaderAbstract) -> None:
        self.obj = obj

    async def __aenter__(self) -> IOAsyncReaderAbstract:
        await self.obj.open()
        return self.obj

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.obj.close()


@dataclass
class IOExpectation:
    start: int
    size: int
    expectation: bytes

    @property
    def name(self) -> str:
        return f"{self.start}-{self.size}"


@dataclass
class IOInterface:
    interface: IOReaderAbstract
    name: str


@pytest.fixture
def local_file(tmp_path: pathlib.Path) -> Generator[pathlib.Path]:
    filepath = tmp_path / "local_file.bin"
    with open(filepath, "bw") as fp:
        fp.write(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    yield filepath
    filepath.unlink()


metal_interface: list[IOInterface] = [
    IOInterface(MmapIO(), "mmap"),
    IOInterface(SeekIO(), "seek"),
]

if PosixPreadIO.is_supported(advice=False):
    metal_interface.append(IOInterface(PosixPreadIO(advice=False), "pread"))
if PosixPreadIO.is_supported(advice=True):
    metal_interface.append(IOInterface(PosixPreadIO(advice=True), "advice"))

metal_expectations = info = [
    IOExpectation(start=0, size=0, expectation=b""),
    IOExpectation(start=0, size=2, expectation=b"AB"),
    IOExpectation(start=0, size=26, expectation=b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    IOExpectation(start=1, size=25, expectation=b"BCDEFGHIJKLMNOPQRSTUVWXYZ"),
    IOExpectation(start=25, size=1, expectation=b"Z"),
    IOExpectation(start=26, size=0, expectation=b""),
    IOExpectation(start=28, size=0, expectation=b""),
]


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.parametrize("info", metal_expectations, ids=lambda info: info.name)
def test_sync_metal_read(local_file: pathlib.Path, metal: IOInterface, info: IOExpectation) -> None:
    reader = metal.interface.sync_reader()
    with AutoSyncOpen(reader(local_file)) as fp:
        assert fp.read(info.start, info.size) == info.expectation


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.parametrize("info", metal_expectations, ids=lambda info: info.name)
@pytest.mark.asyncio
async def test_async_metal_read(local_file: pathlib.Path, metal: IOInterface, info: IOExpectation) -> None:
    reader = metal.interface.async_reader()
    async with AutoAsyncOpen(reader(local_file)) as fp:
        assert await fp.read(info.start, info.size) == info.expectation


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_disallow_short(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.sync_reader()
    with AutoSyncOpen(reader(local_file)) as fp:
        assert fp.read(25, 1) == b"Z"

        with pytest.raises(IndexError, match="shorter than requested size"):
            fp.read(25, 2)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_disallow_short(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.async_reader()
    async with AutoAsyncOpen(reader(local_file)) as fp:
        assert await fp.read(25, 1) == b"Z"

        with pytest.raises(IndexError, match="shorter than requested size"):
            await fp.read(25, 2)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_filesize(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.sync_reader()
    with AutoSyncOpen(reader(local_file)) as fp:
        assert fp.filesize() == 26


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_filesize(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.async_reader()
    async with AutoAsyncOpen(reader(local_file)) as fp:
        assert await fp.filesize() == 26


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_metal_missing(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.sync_reader()
    with pytest.raises(FileNotFoundError):
        with AutoSyncOpen(reader(tmp_path / "test_sync_metal_missing.bin")):
            pass


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_metal_missing(tmp_path: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.async_reader()
    with pytest.raises(FileNotFoundError):
        async with AutoAsyncOpen(reader(tmp_path / "test_async_metal_missing.bin")):
            pass


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
def test_sync_metal_negative_start(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.sync_reader()
    with AutoSyncOpen(reader(local_file)) as fp:
        with pytest.raises(IndexError, match="cannot have negative offset"):
            fp.read(-1, 2)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda metal: metal.name)
@pytest.mark.asyncio
async def test_async_metal_negative_start(local_file: pathlib.Path, metal: IOInterface) -> None:
    reader = metal.interface.async_reader()
    async with AutoAsyncOpen(reader(local_file)) as fp:
        with pytest.raises(IndexError, match="cannot have negative offset"):
            await fp.read(-1, 2)


def test_repeated_posix_pread_close_preserves_reused_file_descriptor(tmp_path: pathlib.Path) -> None:
    if not PosixPreadIO.is_supported():
        pytest.skip("PosixPreadIO is unavailable")
    path = tmp_path / "data.bin"
    path.write_bytes(b"dataset")
    reader = PosixPreadIO().sync_reader()(path)
    reader.open()
    reader.close()
    with path.open("rb") as replacement:
        reader.close()
        assert replacement.read() == b"dataset"
