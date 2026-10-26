from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import NoReturn, overload

import numpy as np
import torch
import pytest

from guideloom.dataset import DocumentDataset, StreamDataset
from guideloom.pytorch import StatefulDataLoader
from guideloom.pytorch.stateful_dataloader import _use_non_blocking_transfer
from guideloom.storage import DocumentStorage, StreamStorage
from guideloom.storage_format import DocumentPickleFormat, StreamNumpyFormat
from guideloom.types import DatasetAbstract, DatasetSyncReaderAbstract, StatefulDataLoaderStateDict


def _dataloader_with_state[SourceT: object, BatchT: object](
    dataloader: StatefulDataLoader[SourceT, BatchT], state: StatefulDataLoaderStateDict
) -> StatefulDataLoader[SourceT, BatchT]:
    dataloader = replace(dataloader)  # this will copy the dataloader, but with a fresh state.
    assert not dataloader._state.active
    assert dataloader._state.observation_offset == 0
    assert dataloader._state.epoch == 0
    dataloader.load_state_dict(state)
    return dataloader


class _SyncMockDataset[OutT](Sequence[OutT]):
    length: int
    fn: Callable[[int], OutT]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[OutT]:
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, index: int) -> OutT: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[OutT]: ...

    def __getitem__(self, index: int | slice) -> OutT | Sequence[OutT]:
        if isinstance(index, int):
            return self.fn(index)
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.fn(i) for i in range(start, stop, step)]
        raise NotImplementedError(f"unsupported input {index}")


class MockDataset[OutT](_SyncMockDataset[OutT], DatasetAbstract[OutT]):
    def __init__(self, fn: Callable[[int], OutT], length: int) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.fn = fn
        self.length = length

    def sync_reader(self) -> "MockSyncReaderDataset[OutT]":
        return MockSyncReaderDataset(self.fn, self.length)

    def async_reader(self) -> NoReturn:
        raise NotImplementedError("not yet implemented")


class MockSyncReaderDataset[OutT](_SyncMockDataset[OutT], DatasetSyncReaderAbstract[OutT]):
    def __init__(self, fn: Callable[[int], OutT], length: int) -> None:
        self.length = length
        self.fn = fn

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


@dataclass
class DistConfig:
    rank: int
    world_size: int
    batch_size: int

    @property
    def name(self) -> str:
        return f"rank-{self.rank}-world-{self.world_size}-batch-{self.batch_size}"

    @staticmethod
    def all_configs(
        max_world_size: int, max_batch_size: int = 1, dataset_length: int = 10
    ) -> "list[DistConfig]":
        configs = []
        for world_size in range(1, max_world_size + 1):
            for batch_size in range(1, max_batch_size + 1):
                if world_size * batch_size <= dataset_length:
                    for rank in range(1, world_size):
                        configs.append(DistConfig(rank, world_size, batch_size))
        return configs


@dataclass
class DropLast:
    value: bool

    def num_batches(self, length: int, batch_size: int) -> int:
        return (length // batch_size) if self.value else _ceil_div(length, batch_size)

    @property
    def name(self) -> str:
        return "last-drop" if self.value else "last-keep"

    @staticmethod
    def all_configs() -> "list[DropLast]":
        return [DropLast(value=True), DropLast(value=False)]


@pytest.fixture
def len_10_hardcoded() -> list[list[int]]:
    return [
        [0, 81, 4, 25, 64, 9, 49, 36, 16, 1],  # epoch 0
        [25, 81, 9, 4, 0, 64, 1, 16, 36, 49],  # epoch 1
        [64, 36, 1, 9, 81, 49, 0, 25, 16, 4],  # epoch 2
        [64, 4, 9, 0, 25, 36, 1, 16, 81, 49],  # epoch 3
        [9, 64, 49, 4, 0, 16, 1, 25, 81, 36],  # epoch 4
        [64, 0, 36, 1, 9, 25, 4, 16, 49, 81],  # epoch 5
        [4, 9, 25, 36, 16, 81, 1, 49, 64, 0],  # epoch 6
        [1, 36, 4, 49, 0, 81, 64, 9, 25, 16],  # epoch 7
    ]


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


def test_dataloader_simple(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device
    ) as dataloader:
        assert list(dataloader) == dataset[:]


def test_dataloader_multiple_epochs(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device
    ) as dataloader:
        for _ in range(10):
            assert list(dataloader) == dataset[:]


def test_dataloader_reject_partial_completion(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device
    ) as dataloader:
        iter(dataloader)
        with pytest.raises(RuntimeError):
            iter(dataloader)
        with pytest.raises(RuntimeError):
            dataloader.load_state_dict(dataloader.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("device_idx", range(torch.cuda.device_count()), ids=lambda x: f"cuda-device-{x}")
def test_dataloader_pin_memory(cpu_device: torch.device, device_idx: int) -> None:
    pin_device = torch.device(f"cuda:{device_idx}")
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: torch.tensor(x[0], device=cpu_device),
        local_batch_size=1,
        device=pin_device,
        pin_memory=True,
    ) as dataloader:
        observations = list(dataloader)
        for obs in observations:
            assert obs.device == pin_device
        assert observations == dataset[:]


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.mps.is_available()), reason="Acceleration device is not available"
)
def test_dataloader_move_device(cpu_device: torch.device) -> None:
    if torch.cuda.is_available():
        move_device = torch.device("cuda:0")
    elif torch.mps.is_available():
        move_device = torch.device("mps:0")
    else:
        raise ValueError("unreachable via skipif")

    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: torch.tensor(x[0], device=cpu_device),
        local_batch_size=1,
        device=move_device,
    ) as dataloader:
        observations = list(dataloader)
        for obs in observations:
            assert obs.device == move_device
        assert observations == dataset[:]


def test_use_non_blocking_transfer_only_for_cuda() -> None:
    assert _use_non_blocking_transfer(torch.device("cuda"))
    assert _use_non_blocking_transfer(torch.device("cuda", 1))
    assert not _use_non_blocking_transfer(torch.device("cpu"))
    assert not _use_non_blocking_transfer(torch.device("mps"))


def test_dataloader_cpu_device(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: torch.tensor(x[0], device=cpu_device),
        local_batch_size=1,
        device=cpu_device,
    ) as dataloader:
        observations = list(dataloader)
        for obs in observations:
            assert obs.device == cpu_device
        assert observations == dataset[:]


def test_dataloader_shuffle_hardcoded(len_10_hardcoded: list[list[int]], cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x[0],
        local_batch_size=1,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
    ) as dataloader:
        for epoch_expectation in len_10_hardcoded:
            assert list(dataloader) == epoch_expectation


@pytest.mark.parametrize("step", range(11), ids=lambda x: f"step-{x}")
def test_dataloader_resume_state_one_epoch(step: int, cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device
    ) as dataloader:
        iterator = iter(dataloader)
        items = [next(iterator) for _ in range(step)]
        assert dataset[:step] == items

        state = dataloader.state_dict()
        assert dataset[step:] == list(_dataloader_with_state(dataloader, state))
        assert dataset[step:] == list(iterator)


@pytest.mark.parametrize("step", range(11), ids=lambda x: f"step-{x}")
@pytest.mark.parametrize("epochs", range(8), ids=lambda x: f"epoch-{x}")
def test_dataloader_resume_state_mutple_epochs(step: int, epochs: int, cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device
    ) as dataloader:
        # run epochs
        for _ in range(epochs):
            list(dataloader)

        # run n steps
        iterator = iter(dataloader)
        items = [next(iterator) for _ in range(step)]
        assert dataset[:step] == items

        state = dataloader.state_dict()
        assert dataset[step:] == list(_dataloader_with_state(dataloader, state))
        assert dataset[step:] == list(iterator)


@pytest.mark.parametrize("step", range(11), ids=lambda x: f"step-{x}")
@pytest.mark.parametrize("epochs", range(8), ids=lambda x: f"epoch-{x}")
def test_dataloader_resume_state_mutple_epochs_with_shuffle(
    step: int, epochs: int, len_10_hardcoded: list[list[int]], cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x[0],
        local_batch_size=1,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
    ) as dataloader:
        # run epochs
        for epoch_expected in len_10_hardcoded[:epochs]:
            assert list(dataloader) == epoch_expected

        # run n steps
        iterator = iter(dataloader)
        items = [next(iterator) for _ in range(step)]
        assert len_10_hardcoded[epochs][:step] == items

        state = dataloader.state_dict()
        assert len_10_hardcoded[epochs][step:] == list(_dataloader_with_state(dataloader, state))
        assert len_10_hardcoded[epochs][step:] == list(iterator)


@pytest.mark.parametrize("old_dist_config", DistConfig.all_configs(3, 2), ids=lambda x: x.name)
@pytest.mark.parametrize("dist_config", DistConfig.all_configs(3, 2), ids=lambda x: x.name)
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_resume_with_different_partitioning(
    old_dist_config: DistConfig,
    dist_config: DistConfig,
    drop_last: DropLast,
    cpu_device: torch.device,
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    samples_by_rank: list[list[int]] = []
    state: StatefulDataLoaderStateDict | None = None
    for rank in range(old_dist_config.world_size):
        with StatefulDataLoader(
            dataset,
            collate_fn=list,
            local_batch_size=old_dist_config.batch_size,
            world_size=old_dist_config.world_size,
            rank=rank,
            drop_last=drop_last.value,
            device=cpu_device,
        ) as loader:
            with iter(loader) as iterator:
                samples_by_rank.append(next(iterator))
                state = loader.state_dict()
    assert state is not None
    observations = [
        sample for samples in zip_longest(*samples_by_rank) for sample in samples if sample is not None
    ]
    offset = state["observation_offset"]

    global_batch_size = dist_config.world_size * dist_config.batch_size
    samples_by_rank = []
    for rank in range(dist_config.world_size):
        with StatefulDataLoader(
            dataset,
            collate_fn=list,
            local_batch_size=dist_config.batch_size,
            world_size=dist_config.world_size,
            rank=rank,
            drop_last=drop_last.value,
            device=cpu_device,
        ) as loader:
            loader.load_state_dict(state)
            samples_by_rank.append([sample for batch in loader for sample in batch])
    observations.extend(
        sample for samples in zip_longest(*samples_by_rank) for sample in samples if sample is not None
    )
    # drop_last truncates the remaining tail to the new global batch size.
    num_batches = drop_last.num_batches(max(0, len(dataset) - offset), global_batch_size)
    assert observations == dataset[: offset + num_batches * global_batch_size]


@pytest.mark.parametrize("batch_size", range(1, 11), ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_batch(batch_size: int, drop_last: DropLast, cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=batch_size,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        assert list(dataloader) == [
            dataset[i * batch_size : (i + 1) * batch_size]
            for i in range(drop_last.num_batches(len(dataset), batch_size))
        ]


@pytest.mark.parametrize("batch_size", range(1, 11), ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_batch_with_shuffle(
    batch_size: int, drop_last: DropLast, len_10_hardcoded: list[list[int]], cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=batch_size,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        for epoch_expectation in len_10_hardcoded:
            assert list(dataloader) == [
                epoch_expectation[i * batch_size : (i + 1) * batch_size]
                for i in range(drop_last.num_batches(len(dataset), batch_size))
            ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(8), ids=lambda x: x.name)
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed(
    dist_config: DistConfig, drop_last: DropLast, cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=1,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        assert list(dataloader) == [
            dataset[
                i * dist_config.world_size + dist_config.rank : (i + 1) * dist_config.world_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(drop_last.num_batches(len(dataset), dist_config.world_size))
        ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(8), ids=lambda x: x.name)
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_shuffle(
    dist_config: DistConfig, drop_last: DropLast, len_10_hardcoded: list[list[int]], cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=1,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        for epoch_expectation in len_10_hardcoded:
            assert list(dataloader) == [
                epoch_expectation[
                    i * dist_config.world_size + dist_config.rank : (i + 1) * dist_config.world_size
                    + dist_config.rank : dist_config.world_size
                ]
                for i in range(drop_last.num_batches(len(dataset), dist_config.world_size))
            ]


def test_dataloader_distributed_shuffle_without_seed(cpu_device: torch.device) -> None:
    # Ranks do not communicate, so each rank must derive the same permutation. With the default
    # shuffle_seed=None, each rank draws its own entropy, so ranks overlap and samples are skipped.
    dataset = MockDataset(lambda i: i, length=100)
    observed: list[int] = []
    for rank in range(2):
        with StatefulDataLoader(
            dataset,
            collate_fn=lambda x: x,
            local_batch_size=1,
            world_size=2,
            rank=rank,
            shuffle=True,
            device=cpu_device,
        ) as dataloader:
            observed.extend(sample for batch in dataloader for sample in batch)

    assert sorted(observed) == list(range(len(dataset)))


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(8, 4), ids=lambda x: x.name)
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_batch_size(
    dist_config: DistConfig, drop_last: DropLast, cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        assert list(dataloader) == [
            dataset[
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(drop_last.num_batches(len(dataset), skip_size))
        ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(8, 4), ids=lambda x: x.name)
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_batch_size_shuffle(
    dist_config: DistConfig, drop_last: DropLast, len_10_hardcoded: list[list[int]], cpu_device: torch.device
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        for epoch_expectation in len_10_hardcoded:
            assert list(dataloader) == [
                epoch_expectation[
                    i * skip_size + dist_config.rank : (i + 1) * skip_size
                    + dist_config.rank : dist_config.world_size
                ]
                for i in range(drop_last.num_batches(len(dataset), skip_size))
            ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(8, 4), ids=lambda x: x.name)
@pytest.mark.parametrize("step", range(2), ids=lambda x: f"step-{x}")
@pytest.mark.parametrize("epochs", range(8), ids=lambda x: f"epoch-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_batch_shuffle_resume_state(
    dist_config: DistConfig,
    step: int,
    epochs: int,
    drop_last: DropLast,
    len_10_hardcoded: list[list[int]],
    cpu_device: torch.device,
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        # run epochs
        for epoch_expectation in len_10_hardcoded[:epochs]:
            assert list(dataloader) == [
                epoch_expectation[
                    i * skip_size + dist_config.rank : (i + 1) * skip_size
                    + dist_config.rank : dist_config.world_size
                ]
                for i in range(drop_last.num_batches(len(dataset), skip_size))
            ]

        # run n steps
        iterator = iter(dataloader)
        items = [v for _, v in zip(range(step), iterator, strict=False)]
        assert items == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step)
        ]

        state = dataloader.state_dict()
        assert list(_dataloader_with_state(dataloader, state)) == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]
        assert list(iterator) == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(2, 3), ids=lambda x: x.name)
@pytest.mark.parametrize("num_workers", range(1, 2), ids=lambda x: f"workers-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_batch_size_shuffle_workers(
    dist_config: DistConfig,
    num_workers: int,
    drop_last: DropLast,
    len_10_hardcoded: list[list[int]],
    cpu_device: torch.device,
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        num_workers=num_workers,
        prefetch=num_workers,
        drop_last=drop_last.value,
    ) as dataloader:
        for epoch_expectation in len_10_hardcoded:
            assert list(dataloader) == [
                epoch_expectation[
                    i * skip_size + dist_config.rank : (i + 1) * skip_size
                    + dist_config.rank : dist_config.world_size
                ]
                for i in range(drop_last.num_batches(len(dataset), skip_size))
            ]


@pytest.mark.parametrize("dist_config", DistConfig.all_configs(2, 3), ids=lambda x: x.name)
@pytest.mark.parametrize("step", range(2), ids=lambda x: f"step-{x}")
@pytest.mark.parametrize("epochs", range(2), ids=lambda x: f"epoch-{x}")
@pytest.mark.parametrize("num_workers", range(1, 2), ids=lambda x: f"workers-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_batch_shuffle_resume_state_workers(
    dist_config: DistConfig,
    step: int,
    epochs: int,
    num_workers: int,
    drop_last: DropLast,
    len_10_hardcoded: list[list[int]],
    cpu_device: torch.device,
) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        shuffle_seed=0,
        device=cpu_device,
        num_workers=num_workers,
        prefetch=num_workers,
        drop_last=drop_last.value,
    ) as dataloader:
        # run epochs
        for epoch_expectation in len_10_hardcoded[:epochs]:
            assert list(dataloader) == [
                epoch_expectation[
                    i * skip_size + dist_config.rank : (i + 1) * skip_size
                    + dist_config.rank : dist_config.world_size
                ]
                for i in range(drop_last.num_batches(len(dataset), skip_size))
            ]

        # run n steps
        iterator = iter(dataloader)
        items = [v for _, v in zip(range(step), iterator, strict=False)]
        assert items == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step)
        ]

        state = dataloader.state_dict()
        assert list(_dataloader_with_state(dataloader, state)) == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]
        assert list(iterator) == [
            len_10_hardcoded[epochs][
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]


@pytest.mark.parametrize(
    "dist_config", DistConfig.all_configs(2, 3, dataset_length=200), ids=lambda x: x.name
)
@pytest.mark.parametrize("step", range(2), ids=lambda x: f"step-{x}")
@pytest.mark.parametrize("num_workers", range(1, 2), ids=lambda x: f"workers-{x}")
@pytest.mark.parametrize("prefetch_factor", range(1, 2), ids=lambda x: f"prefetch-{x}")
@pytest.mark.parametrize("drop_last", DropLast.all_configs(), ids=lambda x: x.name)
def test_dataloader_distributed_resume_state_workers_prefetch(
    dist_config: DistConfig,
    step: int,
    num_workers: int,
    prefetch_factor: int,
    drop_last: DropLast,
    cpu_device: torch.device,
) -> None:
    dataset = MockDataset(lambda i: i**2, length=200)
    skip_size = dist_config.world_size * dist_config.batch_size
    with StatefulDataLoader(
        dataset,
        collate_fn=lambda x: x,
        local_batch_size=dist_config.batch_size,
        world_size=dist_config.world_size,
        rank=dist_config.rank,
        num_workers=num_workers,
        prefetch=num_workers * prefetch_factor,
        device=cpu_device,
        drop_last=drop_last.value,
    ) as dataloader:
        # run n steps
        iterator = iter(dataloader)
        items = [v for _, v in zip(range(step), iterator, strict=False)]
        assert items == [
            dataset[
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step)
        ]

        state = dataloader.state_dict()
        assert list(_dataloader_with_state(dataloader, state)) == [
            dataset[
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]
        assert list(iterator) == [
            dataset[
                i * skip_size + dist_config.rank : (i + 1) * skip_size
                + dist_config.rank : dist_config.world_size
            ]
            for i in range(step, drop_last.num_batches(len(dataset), skip_size))
        ]


def test_not_release_raise_error(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset=dataset, collate_fn=lambda x: x[0], local_batch_size=2, device=cpu_device
    ) as dataloader:
        iter(dataloader)
        with pytest.raises(RuntimeError, match="was not drained or released"):
            dataloader.load_state_dict(dataloader.state_dict())
        with pytest.raises(RuntimeError, match="was not drained or released"):
            iter(dataloader)


def test_release_dataloader(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset=dataset, collate_fn=lambda x: x[0], local_batch_size=2, device=cpu_device
    ) as dataloader:
        with iter(dataloader):
            pass
        dataloader.load_state_dict(dataloader.state_dict())
        iter(dataloader)


def test_world_size_less_than_1(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with pytest.raises(ValueError, match="less than 1"):
        with StatefulDataLoader(
            dataset=dataset, collate_fn=lambda x: x[0], local_batch_size=2, world_size=0, device=cpu_device
        ):
            pass


def test_rank_less_than_0(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with pytest.raises(ValueError, match="less than 0"):
        with StatefulDataLoader(
            dataset=dataset, collate_fn=lambda x: x[0], local_batch_size=2, rank=-1, device=cpu_device
        ):
            pass


def test_rank_greater_equal_world_size(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with pytest.raises(ValueError, match="world_size"):
        with StatefulDataLoader(
            dataset=dataset,
            collate_fn=lambda x: x[0],
            local_batch_size=2,
            rank=2,
            world_size=2,
            device=cpu_device,
        ):
            pass


def test_prefetch_less_than_num_workers_warns(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with StatefulDataLoader(
        dataset=dataset,
        collate_fn=lambda x: x[0],
        local_batch_size=2,
        num_workers=4,
        prefetch=8,
        device=cpu_device,
    ):
        pass
    with pytest.warns(UserWarning, match="not all workers will be utilized"):
        with StatefulDataLoader(
            dataset=dataset,
            collate_fn=lambda x: x[0],
            local_batch_size=2,
            num_workers=8,
            prefetch=2,
            device=cpu_device,
        ):
            pass


def test_pin_memory_to_non_cuda_device(cpu_device: torch.device) -> None:
    dataset = MockDataset(lambda i: i**2, length=10)
    with pytest.raises(ValueError, match="pin_memory"):
        with StatefulDataLoader(
            dataset=dataset, collate_fn=lambda x: x[0], local_batch_size=1, device=cpu_device, pin_memory=True
        ):
            pass


@pytest.mark.parametrize("rank", range(2), ids=lambda x: f"rank-{x}")
def test_valid_dataset_size(rank: int, cpu_device: torch.device) -> None:
    with StatefulDataLoader(
        dataset=MockDataset(lambda i: i**2, length=8),
        collate_fn=lambda x: x[0],
        rank=rank,
        world_size=2,
        local_batch_size=4,
        device=cpu_device,
    ):
        pass

    with pytest.raises(ValueError, match="dataset has fewer elements"):
        with StatefulDataLoader(
            dataset=MockDataset(lambda i: i**2, length=7),
            collate_fn=lambda x: x[0],
            rank=rank,
            world_size=2,
            local_batch_size=4,
            device=cpu_device,
        ):
            pass


def _collate_tokens(samples: Sequence[Mapping[str, np.ndarray]]) -> torch.Tensor:
    return torch.from_numpy(np.stack([sample["tokens"] for sample in samples]))


def test_from_storage_stream(tmp_path: Path, cpu_device: torch.device) -> None:
    storage_format = StreamNumpyFormat(np.dtype([("tokens", "<i4")]))
    storage = StreamStorage(tmp_path, storage_format)
    with storage.sync_writer() as writer:
        writer.append(0, {"tokens": np.arange(8, dtype=np.int32)})

    with StatefulDataLoader.from_storage(
        storage,
        _collate_tokens,
        sequence_length=4,
        device=cpu_device,
        local_batch_size=2,
        shuffle_seed=7,
    ) as loader:
        assert isinstance(loader.dataset, StreamDataset)
        assert loader.shuffle_seed == 7
        assert loader.local_batch_size == 2
        batches = list(loader)

    assert len(batches) == 1
    assert batches[0].shape == (2, 4)
    assert batches[0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_from_storage_document(tmp_path: Path, cpu_device: torch.device) -> None:
    storage = DocumentStorage(tmp_path, DocumentPickleFormat[dict[str, int]]())
    with storage.sync_writer() as writer:
        for value in range(4):
            writer.append(0, {"value": value})

    with StatefulDataLoader.from_storage(
        storage,
        lambda samples: [sample["value"] for sample in samples],
        device=cpu_device,
        local_batch_size=2,
    ) as loader:
        assert isinstance(loader.dataset, DocumentDataset)
        assert list(loader) == [[0, 1], [2, 3]]


def test_from_storage_stream_requires_sequence_length(tmp_path: Path) -> None:
    storage = StreamStorage(tmp_path, StreamNumpyFormat(np.dtype([("tokens", "<i4")])))
    with pytest.raises(ValueError, match="sequence_length"):
        StatefulDataLoader.from_storage(storage, lambda samples: samples)


def test_from_storage_document_rejects_sequence_length(tmp_path: Path) -> None:
    storage = DocumentStorage(tmp_path, DocumentPickleFormat[dict[str, int]]())
    with pytest.raises(TypeError, match="sequence_length"):
        StatefulDataLoader.from_storage(
            storage,
            lambda samples: samples,
            sequence_length=2,
        )


def test_from_storage_rejects_unsupported_storage() -> None:
    with pytest.raises(TypeError, match="unsupported storage type"):
        StatefulDataLoader.from_storage(object(), lambda samples: samples)  # pyright: ignore[reportArgumentType, reportCallIssue]
