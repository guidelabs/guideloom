from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Literal, NoReturn, cast, final, overload
import warnings

import numpy as np
import torch
import torchdata.nodes as tn

from ..dataset import DocumentDataset, StreamDataset
from ..shuffle import FeistelShuffler
from ..types import (
    DatasetAbstract,
    DatasetSyncReaderAbstract,
    DocumentStorageAbstract,
    StatefulDataLoaderStateDict,
    StreamStorageAbstract,
    SyncContextManager,
)
from ._move_data_to_device import _move_data_to_device
from ._pin_memory import PinMemory


@dataclass(eq=False, slots=True)
class _SharedInternalState:
    epoch: int = 0
    observation_offset: int = 0
    active: bool = False


def _shutdown_torchdata_nodes_chain(source: tn.BaseNode[object]) -> None:
    # because this code can run in a __del__, the programming here is very defensive
    node: tn.BaseNode[object] | None = source
    while node is not None:
        if isinstance(node, tn.ParallelMapper):
            internal = getattr(node, "_it", None)
            if isinstance(internal, tn.Unbatcher):
                internal = getattr(internal, "source", None)
            # internal should be _ParallelMapperImpl
            nested = getattr(internal, "_it", None)
            if nested is not None:
                internal = nested
            shutdown = getattr(internal, "_shutdown", None)
            if callable(shutdown):
                shutdown()
        elif isinstance(node, tn.Prefetcher | tn.PinMemory | PinMemory):
            internal = getattr(node, "_it", None)
            shutdown = getattr(internal, "_shutdown", None)
            if callable(shutdown):
                shutdown()

        if isinstance(
            node, tn.ParallelMapper | tn.Prefetcher | tn.PinMemory | PinMemory | tn.Batcher | tn.Unbatcher
        ):
            node = getattr(node, "source", None)
        else:
            node = None


@final
class _SyncIteratorState[T](Iterator[T], SyncContextManager):
    __slots__ = ("_global_batch_size", "_released", "_source", "_state")

    def __init__(self, source: tn.BaseNode[T], state: _SharedInternalState, global_batch_size: int) -> None:
        self._source = source
        self._state = state
        self._global_batch_size = global_batch_size
        self._released = False

    def open(self) -> None:
        pass

    def close(self, *, increment_epoch: bool = False) -> None:
        self._released = True
        self._state.epoch += int(increment_epoch)
        self._state.observation_offset = 0
        self._state.active = False
        _shutdown_torchdata_nodes_chain(self._source)

    def __next__(self) -> T:
        if self._released:
            raise RuntimeError("the iter(SequenceDataLoader) was already released.")

        try:
            val = next(self._source)
            self._state.observation_offset += self._global_batch_size
        except StopIteration:
            self.close(increment_epoch=True)
            raise
        else:
            return val


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _use_non_blocking_transfer(device: torch.device) -> bool:
    # CUDA orders a host-to-device copy before later work on the same stream, so
    # the host can queue it. MPS and other accelerators do not make that guarantee.
    return device.type == "cuda"


class _MapTailIndicesToThemselves:
    def __init__(self, mapper: FeistelShuffler, length: int) -> None:
        self.mapper = mapper
        self.length = length

    def __call__(self, index: int) -> int:
        if index < self.length:
            return self.mapper.shuffle(index)
        return index


@dataclass
class _OOBSentinel:
    pass


class _MapOOBIndicesToSentinel[T: object]:
    def __init__(self, mapper: Sequence[T], length: int) -> None:
        self.mapper = mapper
        self.length = length

    def __call__(self, index: int) -> T | _OOBSentinel:
        if index < self.length:
            return self.mapper[index]
        return _OOBSentinel()


class _callFnWithoutSentinel[SourceT: object, BatchT: object]:
    def __init__(self, fn: Callable[[Sequence[SourceT]], BatchT]) -> None:
        self.fn = fn

    def __call__(self, batch: Sequence[SourceT | _OOBSentinel]) -> BatchT:
        return self.fn([obs for obs in batch if not isinstance(obs, _OOBSentinel)])


# Not a problem if things takes a snapshot. However, we don't use it for anything,
# so might as well disable it.
NEVER_SNAPSHOT_INTERVAL = 2**42  # 4398 billion


@final
@dataclass(eq=False, repr=False, frozen=True)
class StatefulDataLoader[SourceT: object, BatchT: object](Iterable[BatchT], Sized, SyncContextManager):
    """Stateful iterable loader that partitions a dataset across ranks.

    Use the loader as a context manager so its dataset reader is opened and closed. Exhaust an
    iterator, or close it, before starting another iterator on the same loader.

    Args:
        dataset (DatasetAbstract[SourceT]): Dataset to load from.
        collate_fn (Callable[[Sequence[SourceT]], BatchT]): Function that collates samples into a batch.
        device (torch.device): Device batches are copied to. Defaults to the current CUDA device when
            CUDA is available, otherwise CPU. Copies onto CUDA use ``non_blocking=True``.
        local_batch_size (int): Samples per batch for this rank. Defaults to 1.
        rank (int): This rank's index in the distributed job. Defaults to 0.
        world_size (int): Number of ranks. Defaults to 1.
        num_workers (int): Worker threads or processes. Defaults to 0, which does not load batches
            concurrently.
        parallel_method ('thread', 'process'): How `collate_fn` is parallelized. Defaults to thread.
        shuffle (bool): Whether to shuffle sample indexes. Defaults to False.
        shuffle_seed (int | None): Seed for shuffling. Required when `shuffle` is True, and must be
            the same on every rank. Defaults to None. Ignored when `shuffle` is False.
        pin_memory (bool): Whether to pin CPU memory before a CUDA copy. Defaults to False.
        drop_last (bool): Whether to drop the last global batch when it is smaller than
            `local_batch_size * world_size`. Defaults to True. When False and `world_size > 1`, a
            rank can receive an empty sample list, `collate_fn([])`, so every rank takes the same
            number of steps and can call a collective such as `fabric.barrier()`. `collate_fn` must
            accept that empty list. When `world_size == 1`, `collate_fn` never receives an empty list.
        prefetch (int): Batches to prefetch. Defaults to 0.
    """

    dataset: DatasetAbstract[SourceT]
    collate_fn: Callable[[Sequence[SourceT]], BatchT]
    device: torch.device = field(default_factory=_get_default_device, kw_only=True)
    local_batch_size: int = field(default=1, kw_only=True)
    rank: int = field(default=0, kw_only=True)
    world_size: int = field(default=1, kw_only=True)
    num_workers: int = field(default=0, kw_only=True)
    parallel_method: Literal["thread", "process"] = field(default="thread", kw_only=True)
    shuffle: bool = field(default=False, kw_only=True)
    shuffle_seed: int | None = field(default=None, kw_only=True)
    pin_memory: bool = field(default=False, kw_only=True)
    drop_last: bool = field(default=True, kw_only=True)
    prefetch: int = field(default=0, kw_only=True)

    _state: _SharedInternalState = field(init=False, default_factory=_SharedInternalState)

    @overload
    @classmethod
    def from_storage[InT, OutT: object, cBatchT: object](
        cls: type["StatefulDataLoader[OutT, cBatchT]"],
        storage: StreamStorageAbstract[InT, OutT],
        collate_fn: Callable[[Sequence[OutT]], cBatchT],
        *,
        sequence_length: int,
        device: torch.device | None = None,
        local_batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        parallel_method: Literal["thread", "process"] = "thread",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch: int = 0,
    ) -> "StatefulDataLoader[OutT, cBatchT]": ...

    @overload
    @classmethod
    def from_storage[InT, OutT: object, cBatchT: object](
        cls: type["StatefulDataLoader[OutT, cBatchT]"],
        storage: StreamStorageAbstract[InT, OutT],
        collate_fn: Callable[[Sequence[OutT]], cBatchT],
        *,
        sequence_length: None = None,
        device: torch.device | None = None,
        local_batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        parallel_method: Literal["thread", "process"] = "thread",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch: int = 0,
    ) -> NoReturn: ...

    @overload
    @classmethod
    def from_storage[InT, OutT: object, cBatchT: object](
        cls: type["StatefulDataLoader[OutT, cBatchT]"],
        storage: DocumentStorageAbstract[InT, OutT],
        collate_fn: Callable[[Sequence[OutT]], cBatchT],
        *,
        sequence_length: None = None,
        device: torch.device | None = None,
        local_batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        parallel_method: Literal["thread", "process"] = "thread",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch: int = 0,
    ) -> "StatefulDataLoader[OutT, cBatchT]": ...

    @overload
    @classmethod
    def from_storage[InT, OutT: object, cBatchT: object](
        cls: type["StatefulDataLoader[OutT, cBatchT]"],
        storage: DocumentStorageAbstract[InT, OutT],
        collate_fn: Callable[[Sequence[OutT]], cBatchT],
        *,
        sequence_length: int,
        device: torch.device | None = None,
        local_batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        parallel_method: Literal["thread", "process"] = "thread",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch: int = 0,
    ) -> NoReturn: ...

    @classmethod
    def from_storage[InT, OutT: object, cBatchT: object](
        cls: type["StatefulDataLoader[OutT, cBatchT]"],
        storage: StreamStorageAbstract[InT, OutT] | DocumentStorageAbstract[InT, OutT],
        collate_fn: Callable[[Sequence[OutT]], cBatchT],
        *,
        sequence_length: int | None = None,
        device: torch.device | None = None,
        local_batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        parallel_method: Literal["thread", "process"] = "thread",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch: int = 0,
    ) -> "StatefulDataLoader[OutT, cBatchT]":
        """Create a loader by wrapping storage in the matching dataset.

        Stream storage, including streams with metadata, is wrapped in `StreamDataset` and requires
        `sequence_length`. Document storage is wrapped in `DocumentDataset`. The remaining arguments
        match the constructor.

        Args:
            storage (StreamStorageAbstract[InT, OutT] | DocumentStorageAbstract[InT, OutT]): Storage
                to read from.
            collate_fn (Callable[[Sequence[OutT]], cBatchT]): Function to collate samples into a batch.
            sequence_length (int | None): Stream elements per stream sample. Required for stream
                storage and omitted for document storage. Defaults to None.
            device (torch.device | None): Target device for data placement. When omitted, uses CPU or
                the first CUDA device. Defaults to None.
            local_batch_size (int): Number of samples per batch for this rank. Defaults to 1.
            rank (int): Local rank, used for data partitioning. Defaults to 0.
            world_size (int): Total number of ranks. Defaults to 1.
            num_workers (int): Number of worker threads or processes. Defaults to 0.
            parallel_method ('thread', 'process'): Parallelization strategy. Defaults to thread.
            shuffle (bool): Whether to shuffle the dataset. Defaults to False.
            shuffle_seed (int | None): Seed for shuffling. Required when `shuffle` is True.
                Defaults to None.
            pin_memory (bool): Whether to use pinned memory for CUDA. Defaults to False.
            drop_last (bool): Whether to drop the last incomplete global batch. Defaults to True.
            prefetch (int): Number of batches to prefetch. Defaults to 0.

        Returns:
            StatefulDataLoader[OutT, cBatchT]: Loader over the dataset derived from `storage`.
        """
        dataset: DatasetAbstract[OutT]
        match (storage, sequence_length):
            case (StreamStorageAbstract() as storage, int() as sequence_length):
                dataset = StreamDataset(storage, sequence_length)
            case (StreamStorageAbstract(), None):
                raise ValueError("sequence_length must be specified for a stream storage")
            case (DocumentStorageAbstract() as storage, None):
                dataset = DocumentDataset(storage)
            case (DocumentStorageAbstract(), _):
                raise TypeError("sequence_length cannot be specified for a document storage")
            case _:
                raise TypeError(f"unsupported storage type {type(storage).__name__}")

        return cls(
            dataset,
            collate_fn,
            device=_get_default_device() if device is None else device,
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            parallel_method=parallel_method,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch=prefetch,
        )

    @staticmethod
    def get_default_device() -> torch.device:
        """Return the current CUDA device, or CPU when CUDA is unavailable.

        Returns:
            torch.device: `cuda` on `torch.cuda.current_device()` when CUDA is available, otherwise CPU.
        """
        return _get_default_device()

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.local_batch_size

    @cached_property
    def dataset_size(self) -> int:
        return len(self._reader)

    @cached_property
    def _reader(self) -> DatasetSyncReaderAbstract[SourceT]:
        return self.dataset.sync_reader()

    def open(self) -> None:
        """Open the dataset reader.

        Raises:
            ValueError: If the dataset contains fewer observations than `world_size * local_batch_size`.
        """
        self._reader.open()

        if self.dataset_size < self.global_batch_size:
            raise ValueError(
                f"The dataset has fewer elements ({self.dataset_size})"
                f" than the global batch-size ({self.global_batch_size})"
            )

    def close(self) -> None:
        """Close the dataset reader."""
        self._reader.close()

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"cannot have world_size ({self.world_size}) less than 1")

        if self.rank < 0:
            raise ValueError(f"cannot have rank ({self.rank}) less than 0")

        if self.rank >= self.world_size:
            raise ValueError(
                f"cannot have rank ({self.rank}) greater or equal to the world_size ({self.world_size})"
            )

        # Ranks do not communicate, so they only agree on a permutation through a shared seed.
        if self.shuffle and self.shuffle_seed is None:
            raise ValueError(
                "shuffle_seed must be set when shuffle is enabled, to the same value on every rank"
            )

        if self.prefetch * self.local_batch_size < self.num_workers:
            warnings.warn(
                f"the number of observations prefetched ({self.prefetch} * {self.local_batch_size}"
                f" = {self.prefetch * self.local_batch_size}) is less than the number of workers"
                f" ({self.num_workers}), therefore not all workers will be utilized.",
                UserWarning,
                stacklevel=1,
            )

        if self.pin_memory and self.device.type != "cuda":
            raise ValueError(f"pin_memory is set, but the device {self.device} is not CUDA")

        if self.device.type == "cuda" and not self.pin_memory:
            warnings.warn(
                f"device is cuda ({self.device}) however, pin-memory is disabled, you should"
                " almost certainly enable it.",
                UserWarning,
                stacklevel=1,
            )

    def _get_epoch_rng(self, epoch: int) -> np.random.Generator:
        bit_gen = np.random.Philox(np.random.SeedSequence(self.shuffle_seed, spawn_key=(epoch,)))
        return np.random.Generator(bit_gen)

    def __iter__(self) -> _SyncIteratorState[BatchT]:
        if self._state.active:
            raise RuntimeError(
                "cannot call iter(SequenceDataLoader)"
                " if the previouse iterator was not drained or released."
                " You may call iter(SequenceDataLoader).release() to release it."
            )
        self._state.active = True

        # NOTE for drop_last=False, just `drop_last=False` in `tn.Batcher` is not sufficient in a
        # distributed setting, as it could give different number of iterations for each worker, which
        # would cause problem when using communication primitives (e.g. barrier). Instead, the strategy
        # is to extend the indices, such observations outside of the range are included, and then map
        # these observations to a sentinel object, and then remove these sentinel objects before calling
        # `self.collate_fn`.

        # Resume from a global observation offset, independent of the previous batch/world size.
        # Round the remaining length to whole global batches relative to that offset, since it
        # need not align with the current global batch size. With drop_last=False, pad with OOB
        # indices so that all ranks still yield the same number of batches.
        offset = self._state.observation_offset
        remaining = max(0, self.dataset_size - offset)
        if self.drop_last:
            num_batches_remaining = remaining // self.global_batch_size
        else:
            num_batches_remaining = -(-remaining // self.global_batch_size)
        indices = tn.IterableWrapper(
            range(
                offset + self.rank,
                offset + num_batches_remaining * self.global_batch_size,
                self.world_size,
            )
        )

        # apply shuffleing if requested
        if self.shuffle:
            shuffler = FeistelShuffler(self.dataset_size, rng=self._get_epoch_rng(self._state.epoch))
            indices = tn.ParallelMapper(
                indices,
                shuffler.shuffle
                if self.drop_last
                else _MapTailIndicesToThemselves(mapper=shuffler, length=self.dataset_size),
                method="thread",
                num_workers=self.num_workers,
                in_order=True,
                snapshot_frequency=NEVER_SNAPSHOT_INTERVAL,
            )

        # get observations and batch them
        observations = tn.ParallelMapper(
            indices,
            self._reader.__getitem__
            if self.drop_last
            else _MapOOBIndicesToSentinel(mapper=self._reader, length=self.dataset_size),
            method="thread",
            num_workers=self.num_workers,
            in_order=True,
            snapshot_frequency=NEVER_SNAPSHOT_INTERVAL,
        )
        batches = tn.Batcher(observations, batch_size=self.local_batch_size, drop_last=False)
        # When drop_last is set, batches contain no sentinel and collate_fn can be used directly.
        collate: Callable[[Sequence[SourceT | _OOBSentinel]], BatchT]
        if self.drop_last:
            collate = cast("Callable[[Sequence[SourceT | _OOBSentinel]], BatchT]", self.collate_fn)
        else:
            collate = _callFnWithoutSentinel(fn=self.collate_fn)
        collated = tn.ParallelMapper(
            batches,
            collate,
            method=self.parallel_method,
            num_workers=self.num_workers,
            in_order=True,
            snapshot_frequency=NEVER_SNAPSHOT_INTERVAL,
        )

        # apply pin-memory if requested
        if self.pin_memory:
            # we know the device is cuda, because of the __post_init__. Just make sure
            # torch.cuda.current_device() points to the provided device.
            with torch.cuda.device(self.device):
                collated = PinMemory(collated, snapshot_frequency=NEVER_SNAPSHOT_INTERVAL)

        # prefetch observations
        if self.prefetch > 0:
            collated = tn.Prefetcher(collated, prefetch_factor=self.prefetch)

        # if device is CPU, there is no need to move it.
        if self.device.type != "cpu":
            collated = tn.ParallelMapper(
                collated,
                partial(
                    _move_data_to_device,
                    device=self.device,
                    non_blocking=_use_non_blocking_transfer(self.device),
                ),
                method="thread",
                num_workers=0,
                in_order=True,
                snapshot_frequency=NEVER_SNAPSHOT_INTERVAL,
            )

        # state is entirely managed by the `range()` config and `_get_epoch_rng()`. However,
        # torchdata requires .reset to be called.
        collated.reset(initial_state=None)

        # Syncronize the steps and epochs with the self._state variable. Note that this must
        # come after prefetch to match the actually recived content.
        return _SyncIteratorState(collated, self._state, self.global_batch_size)

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.global_batch_size
        # math.ceil(self.dataset_size / self.global_batch_size) but for integers
        return -(-self.dataset_size // self.global_batch_size)  # noqa: PLE0303

    def load_state_dict(self, state_dict: StatefulDataLoaderStateDict) -> None:
        """Restore `epoch` and `observation_offset` from a checkpoint.

        Call this before starting an iterator. A warning is emitted when `shuffle`, `shuffle_seed`,
        or `drop_last` differ from this loader. Those saved values are not applied.
        The global observation offset allows resuming with a different batch size or world size.

        Args:
            state_dict (StatefulDataLoaderStateDict): Checkpoint from `state_dict`.

        Raises:
            RuntimeError: If an iterator on this loader is still active.
        """
        if self._state.active:
            raise RuntimeError(
                "cannot call SequenceDataLoader.load_state_dict()"
                " if the previouse iterator was not drained or released."
                " You may call iter(SequenceDataLoader).release() to release it."
            )

        if self.shuffle != state_dict["shuffle"]:
            warnings.warn(
                f"Loaded shuffle value ({state_dict['shuffle']}) and"
                f" initalized shuffle value ({self.shuffle}) doesn't match."
                " Cannot fully restore state, likely create unexpected results.",
                UserWarning,
                stacklevel=1,
            )

        if self.shuffle_seed != state_dict["shuffle_seed"]:
            warnings.warn(
                f"Loaded shuffle_seed value ({state_dict['shuffle_seed']}) and"
                f" initalized shuffle_seed value ({self.shuffle_seed}) doesn't match."
                " Cannot fully restore state, likely create unexpected results.",
                UserWarning,
                stacklevel=1,
            )

        if self.drop_last != state_dict["drop_last"]:
            warnings.warn(
                f"Loaded drop_last value ({state_dict['drop_last']}) and"
                f" initalized drop_last value ({self.drop_last}) doesn't match."
                " Cannot fully restore state, likely create unexpected results.",
                UserWarning,
                stacklevel=1,
            )

        self._state.epoch = state_dict["epoch"]
        self._state.observation_offset = state_dict["observation_offset"]

    def state_dict(self) -> StatefulDataLoaderStateDict:
        """Return the checkpoint for batches already delivered to the consumer.

        Prefetched batches that have not been yielded do not advance `observation_offset`.
        Each yielded batch advances the offset by `global_batch_size`, including padding in an
        incomplete final batch. The result contains `epoch`, `observation_offset`, `shuffle`,
        `shuffle_seed`, and `drop_last`.

        Returns:
            StatefulDataLoaderStateDict: Loader checkpoint.
        """
        return {
            "epoch": self._state.epoch,
            "observation_offset": self._state.observation_offset,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "drop_last": self.drop_last,
        }
