from collections.abc import Generator, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict
import shutil
import pathlib
import concurrent.futures

import numpy as np
import pytest

from guideloom.dataset import StreamDataset
from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.storage import StreamWithMetaStorage
from guideloom.types import (
    DatasetAsyncReaderAbstract,
    DatasetSyncReaderAbstract,
    IOReaderAbstract,
    StorageBufferTypes,
    StreamWithMetaStorageFormat,
    UserBufferTypes,
    Vector,
)


#
# The MockStorageFormat applied thoughout this test, fairly basic. But should cover everything.
#
class DocumentData[InputDType: np.integer](TypedDict):
    doc_id: int
    stream_data: Vector[InputDType]
    document_ids: Vector[np.int32]
    document_data: Vector[InputDType]


class StreamingData[InputDType: np.generic](TypedDict):
    shard_ids: Vector[np.int32]
    doc_ids: Vector[np.int32]
    document_data: Vector[InputDType]
    stream_data: Vector[InputDType]


class MockStorageFormat[DataDType: np.integer](
    StreamWithMetaStorageFormat[DocumentData[DataDType], StreamingData[DataDType], DataDType]
):
    def size(self, data: DocumentData[DataDType]) -> int:
        return data["stream_data"].size

    def pack(
        self, data: DocumentData[DataDType]
    ) -> Generator[tuple[Hashable, UserBufferTypes | None], None, Vector[DataDType]]:
        for document_id, document_val in zip(
            data["document_ids"].tolist(), data["document_data"].tolist(), strict=True
        ):
            yield ((document_id,), f"{data['doc_id']}-{document_val}".encode())

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
            "shard_ids": np.fromiter((document_shard[i.item()] for i in document_index), dtype=np.int32),
            "doc_ids": np.fromiter((int(document_data_unpack[i][0]) for i in document_index), dtype=np.int32),
            "document_data": np.fromiter(
                (int(document_data_unpack[i][1]) for i in document_index), dtype=self.stream_dtype
            ),
            "stream_data": stream_data,
        }


class RandomDatasetInfo(TypedDict):
    dataset_path: pathlib.Path
    stream_data: Vector[np.int32]
    doc_ids: Vector[np.int32]
    shard_ids: Vector[np.int32]
    document_ids: Vector[np.int32]
    document_data: Vector[np.int32]


@pytest.fixture(scope="session")
def random_20k_dataset(tmp_path_factory: pytest.TempPathFactory) -> Generator[RandomDatasetInfo]:
    # Create a shared temporary directory
    tmp_dir = tmp_path_factory.mktemp("random_20k_dataset")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(tmp_dir, storage_format)

    rng = np.random.default_rng(0)
    stream_data: Vector[np.int32] = rng.integers(low=0, high=100, size=20000, dtype=np.int32)
    doc_ids: Vector[np.int32] = np.repeat(np.arange(1000, dtype=np.int32), 20)
    shard_ids: Vector[np.int32] = np.repeat(np.arange(10, dtype=np.int32), 2000)
    document_ids: Vector[np.int32] = np.repeat(np.arange(10000, dtype=np.int32), 2)
    document_data: Vector[np.int32] = np.repeat(rng.integers(low=0, high=100, size=10000, dtype=np.int32), 2)

    assert len(stream_data) == 20000
    assert len(doc_ids) == 20000
    assert len(shard_ids) == 20000
    assert len(document_ids) == 20000
    assert len(document_data) == 20000

    with storage.sync_writer() as writer:
        offset = 0
        for shard in range(10):
            for _ in range(100):
                writer.append(
                    shard,
                    {
                        "doc_id": doc_ids[offset].item(),
                        "stream_data": stream_data[offset : offset + 20],
                        "document_ids": document_ids[offset : offset + 20],
                        "document_data": document_data[offset : offset + 20],
                    },
                )
                offset += 20

    # Yield path to dataset to the tests
    yield {
        "dataset_path": tmp_dir,
        "stream_data": stream_data,
        "doc_ids": doc_ids,
        "shard_ids": shard_ids,
        "document_ids": document_ids,
        "document_data": document_data,
    }

    # Teardown logic: remove temp directory
    shutil.rmtree(tmp_dir)


def _assert_read_match_dataset[DataDType: np.integer](
    read: StreamingData[DataDType], ref: RandomDatasetInfo, index: int, chunk_size: int
) -> None:
    global_stream_data = ref["stream_data"]
    global_doc_ids = ref["doc_ids"]
    global_shard_ids = ref["shard_ids"]
    global_document_data = ref["document_data"]

    np.testing.assert_equal(
        read["shard_ids"], global_shard_ids[index * chunk_size : (index + 1) * chunk_size]
    )
    np.testing.assert_equal(read["doc_ids"], global_doc_ids[index * chunk_size : (index + 1) * chunk_size])
    np.testing.assert_equal(
        read["stream_data"], global_stream_data[index * chunk_size : (index + 1) * chunk_size]
    )
    np.testing.assert_equal(
        read["document_data"], global_document_data[index * chunk_size : (index + 1) * chunk_size]
    )


@dataclass
class DatasetTestConfig:
    chunk_size: int
    check_every: int

    @property
    def name(self) -> str:
        return f"chunk-{self.chunk_size}-every-{self.check_every}"


test_configs = [
    DatasetTestConfig(1, 100),
    DatasetTestConfig(10, 50),
    DatasetTestConfig(20, 40),
    DatasetTestConfig(100, 10),
    DatasetTestConfig(1000, 1),
    DatasetTestConfig(2000, 1),
    DatasetTestConfig(4000, 1),
    DatasetTestConfig(8000, 1),
]


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
class StreamDatasetConfig:
    constructor: type[StreamDataset[Any]]
    name: str

    def create_sync_dataset[OutDataT: Mapping[str, object]](
        self, storage: StreamWithMetaStorage[Any, OutDataT, Any], chunk_size: int
    ) -> DatasetSyncReaderAbstract[OutDataT]:
        return self.constructor(storage, chunk_size).sync_reader()

    def create_async_dataset[OutDataT: Mapping[str, object]](
        self, storage: StreamWithMetaStorage[Any, OutDataT, Any], chunk_size: int
    ) -> DatasetAsyncReaderAbstract[OutDataT]:
        return self.constructor(storage, chunk_size).async_reader()


dataset_configs = [StreamDatasetConfig(StreamDataset, "sync")]


@pytest.mark.parametrize("info", test_configs, ids=lambda x: x.name)
@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_serial(
    info: DatasetTestConfig,
    metal: IOInterface,
    dataset_config: StreamDatasetConfig,
    random_20k_dataset: RandomDatasetInfo,
) -> None:
    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage, chunk_size=info.chunk_size) as dataset:
        assert len(dataset) == 20000 // info.chunk_size
        for dataset_i in range(0, 20000 // info.chunk_size, info.check_every):
            _assert_read_match_dataset(dataset[dataset_i], random_20k_dataset, dataset_i, info.chunk_size)


@pytest.mark.parametrize("info", test_configs, ids=lambda x: x.name)
@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
@pytest.mark.asyncio
async def test_async_dataset_read_match_serial(
    info: DatasetTestConfig,
    metal: IOInterface,
    dataset_config: StreamDatasetConfig,
    random_20k_dataset: RandomDatasetInfo,
) -> None:
    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    async with dataset_config.create_async_dataset(storage, chunk_size=info.chunk_size) as dataset:
        assert len(dataset) == 20000 // info.chunk_size
        for dataset_i in range(0, 20000 // info.chunk_size, info.check_every):
            _assert_read_match_dataset(
                await dataset[dataset_i], random_20k_dataset, dataset_i, info.chunk_size
            )


@pytest.mark.parametrize("info", test_configs, ids=lambda x: x.name)
@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_thread_concurrency(
    info: DatasetTestConfig,
    metal: IOInterface,
    dataset_config: StreamDatasetConfig,
    random_20k_dataset: RandomDatasetInfo,
) -> None:
    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage, chunk_size=info.chunk_size) as dataset:
        assert len(dataset) == 20000 // info.chunk_size

        indices = range(0, 20000 // info.chunk_size, info.check_every)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            chunks = executor.map(lambda idx: dataset[idx], indices)

        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i, info.chunk_size)


@pytest.mark.parametrize("info", test_configs, ids=lambda x: x.name)
@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
def test_sync_dataset_read_match_many(
    info: DatasetTestConfig,
    metal: IOInterface,
    dataset_config: StreamDatasetConfig,
    random_20k_dataset: RandomDatasetInfo,
) -> None:
    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    with dataset_config.create_sync_dataset(storage, chunk_size=info.chunk_size) as dataset:
        assert len(dataset) == 20000 // info.chunk_size
        indices = range(0, 20000 // info.chunk_size, info.check_every)
        chunks = dataset[0 : (20000 // info.chunk_size) : info.check_every]
        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i, info.chunk_size)


@pytest.mark.parametrize("info", test_configs, ids=lambda x: x.name)
@pytest.mark.parametrize("metal", metal_interface, ids=lambda x: x.name)
@pytest.mark.parametrize("dataset_config", dataset_configs, ids=lambda x: x.name)
@pytest.mark.asyncio
async def test_async_dataset_read_match_many(
    info: DatasetTestConfig,
    metal: IOInterface,
    dataset_config: StreamDatasetConfig,
    random_20k_dataset: RandomDatasetInfo,
) -> None:
    storage_format = MockStorageFormat(stream_dtype=np.dtype(np.int32))
    storage = StreamWithMetaStorage(
        random_20k_dataset["dataset_path"], storage_format, read_interface=metal.interface
    )

    async with dataset_config.create_async_dataset(storage, chunk_size=info.chunk_size) as dataset:
        assert len(dataset) == 20000 // info.chunk_size
        indices = range(0, 20000 // info.chunk_size, info.check_every)
        chunks = await dataset[0 : (20000 // info.chunk_size) : info.check_every]
        for chunk, dataset_i in zip(chunks, indices, strict=True):
            _assert_read_match_dataset(chunk, random_20k_dataset, dataset_i, info.chunk_size)
