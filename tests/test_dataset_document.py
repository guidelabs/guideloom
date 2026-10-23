from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict
import shutil
import struct
import pathlib
import concurrent.futures

import numpy as np
import pytest

from guideloom.dataset import DocumentDataset
from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.storage import DocumentStorage
from guideloom.types import (
    DatasetAsyncReaderAbstract,
    DatasetSyncReaderAbstract,
    DocumentStorageFormat,
    IOReaderAbstract,
    StorageBufferTypes,
    UserBufferTypes,
    Vector,
)


#
# The MockStorageFormat applied thoughout this test, fairly basic. But should cover everything.
#
class InDocumentData(TypedDict):
    doc_id: int
    data: str


class OutDocumentData(TypedDict):
    shard: int
    doc_id: int
    data: str


class MockStorageFormat(DocumentStorageFormat[InDocumentData, OutDocumentData]):
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


class RandomDatasetInfo(TypedDict):
    dataset_path: pathlib.Path
    data: Vector[np.str_]
    doc_ids: Vector[np.int32]
    shard_ids: Vector[np.int32]


@pytest.fixture(scope="session")
def random_20k_dataset(tmp_path_factory: pytest.TempPathFactory) -> Generator[RandomDatasetInfo]:
    # Create a shared temporary directory
    tmp_dir = tmp_path_factory.mktemp("random_20k_dataset")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    storage_format = MockStorageFormat()
    storage = DocumentStorage(tmp_dir, storage_format)

    rng = np.random.default_rng(0)
    data: Vector[np.str_] = rng.integers(low=0, high=100, size=2000, dtype=np.int32).astype(np.dtype("U4"))
    doc_ids: Vector[np.int32] = np.arange(2000, dtype=np.int32)
    shard_ids: Vector[np.int32] = np.repeat(np.arange(10, dtype=np.int32), 200)

    assert len(data) == 2000
    assert len(doc_ids) == 2000
    assert len(shard_ids) == 2000

    with storage.sync_writer() as writer:
        offset = 0
        for shard in range(10):
            for _ in range(200):
                writer.append(shard, {"doc_id": doc_ids[offset].item(), "data": data[offset].item()})
                offset += 1

    # Yield path to dataset to the tests
    yield {"dataset_path": tmp_dir, "data": data, "doc_ids": doc_ids, "shard_ids": shard_ids}

    # Teardown logic: remove temp directory
    shutil.rmtree(tmp_dir)


def _assert_read_match_dataset(read: OutDocumentData, ref: RandomDatasetInfo, index: int) -> None:
    global_data = ref["data"]
    global_doc_ids = ref["doc_ids"]
    global_shard_ids = ref["shard_ids"]

    assert global_data[index] == read["data"]
    assert global_doc_ids[index] == read["doc_id"]
    assert global_shard_ids[index] == read["shard"]


@dataclass
class IOInterface:
    interface: IOReaderAbstract
    name: str


metal_interface: list[IOInterface] = [
    IOInterface(MmapIO(), "mmap"),
    IOInterface(SeekIO(), "seek"),
]

if PosixPreadIO.is_supported(advice=False):
    metal_interface.append(IOInterface(PosixPreadIO(advice=False), "pread"))
if PosixPreadIO.is_supported(advice=True):
    metal_interface.append(IOInterface(PosixPreadIO(advice=True), "advice"))


@dataclass
class DocumentDatasetConfig:
    constructor: type[DocumentDataset[Any]]
    name: str

    def create_sync_dataset[OutDataT: Mapping[str, object]](
        self, storage: DocumentStorage[Any, OutDataT]
    ) -> DatasetSyncReaderAbstract[OutDataT]:
        return self.constructor(storage).sync_reader()

    def create_async_dataset[OutDataT: Mapping[str, object]](
        self, storage: DocumentStorage[Any, OutDataT]
    ) -> DatasetAsyncReaderAbstract[OutDataT]:
        return self.constructor(storage).async_reader()


dataset_configs = [
    DocumentDatasetConfig(DocumentDataset, "sync"),
]


@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_serial(
    metal: IOInterface, dataset_config: DocumentDatasetConfig, random_20k_dataset: RandomDatasetInfo
) -> None:
    storage_format = MockStorageFormat()
    storage = DocumentStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage) as dataset:
        assert len(dataset) == 2000
        for dataset_i in range(2000):
            _assert_read_match_dataset(dataset[dataset_i], random_20k_dataset, dataset_i)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
@pytest.mark.asyncio
async def test_async_dataset_read_match_serial(
    metal: IOInterface, dataset_config: DocumentDatasetConfig, random_20k_dataset: RandomDatasetInfo
) -> None:
    storage_format = MockStorageFormat()
    storage = DocumentStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    async with dataset_config.create_async_dataset(storage) as dataset:
        assert len(dataset) == 2000
        for dataset_i in range(2000):
            _assert_read_match_dataset(await dataset[dataset_i], random_20k_dataset, dataset_i)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_thread_concurrency(
    metal: IOInterface, dataset_config: DocumentDatasetConfig, random_20k_dataset: RandomDatasetInfo
) -> None:
    storage_format = MockStorageFormat()
    storage = DocumentStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage) as dataset:
        assert len(dataset) == 2000

        indices = range(2000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            chunks = executor.map(lambda idx: dataset[idx], indices)

        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i)


@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_many(
    metal: IOInterface, dataset_config: DocumentDatasetConfig, random_20k_dataset: RandomDatasetInfo
) -> None:
    storage_format = MockStorageFormat()
    storage = DocumentStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage) as dataset:
        assert len(dataset) == 2000
        indices = range(2000)
        chunks = dataset[0:2000]
        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i)


@pytest.mark.parametrize("metal", metal_interface[1:2], ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
@pytest.mark.asyncio
async def test_async_dataset_read_match_many(
    metal: IOInterface, dataset_config: DocumentDatasetConfig, random_20k_dataset: RandomDatasetInfo
) -> None:
    check_every = 10 if metal.name == "seek" else 1

    storage_format = MockStorageFormat()
    storage = DocumentStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    async with dataset_config.create_async_dataset(storage) as dataset:
        assert len(dataset) == 2000
        indices = range(0, 2000, check_every)
        chunks = await dataset[0:2000:check_every]
        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i)
