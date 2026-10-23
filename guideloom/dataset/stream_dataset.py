from collections.abc import Awaitable, Sequence
from typing import Any, final, overload
import asyncio

from ..types import (
    DatasetAbstract,
    DatasetAsyncReaderAbstract,
    DatasetSyncReaderAbstract,
    StreamStorageAbstract,
)


@final
class StreamDataset[OutDataT](DatasetAbstract[OutDataT]):
    def __init__(self, storage: StreamStorageAbstract[Any, OutDataT], sequence_length: int) -> None:
        """Create a dataset view of stream storage, with one item per fixed-length window.

        Each observation contains `sequence_length` stream elements. The opened view length is
        `total_length // sequence_length`, so a trailing partial window is omitted. Open a reader
        before indexing it.

        Args:
            storage (StreamStorageAbstract[Any, OutDataT]): Storage that provides access to the database.
            sequence_length (int): Number of stream elements, such as tokens, in each observation.
        """
        self._storage = storage
        self._sequence_length = sequence_length

    def sync_reader(self) -> "StreamSyncReaderDataset[OutDataT]":
        return StreamSyncReaderDataset(self._storage, self._sequence_length)

    def async_reader(self) -> "StreamAsyncReaderDataset[OutDataT]":
        return StreamAsyncReaderDataset(self._storage, self._sequence_length)


@final
class StreamSyncReaderDataset[OutDataT](DatasetSyncReaderAbstract[OutDataT]):
    def __init__(self, storage: StreamStorageAbstract[Any, OutDataT], sequence_length: int) -> None:
        self._reader = storage.sync_reader()
        self._sequence_length = sequence_length
        self._observations = 0
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            return

        self._is_open = True
        self._reader.open()
        self._observations = self._reader.total_length // self._sequence_length

    def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False
        self._reader.close()

    def __len__(self) -> int:
        if not self._is_open:
            raise RuntimeError("Dataset.open() was not called")

        return self._observations

    @overload
    def __getitem__(self, index: int) -> OutDataT: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[OutDataT]: ...

    def __getitem__(self, index: int | slice) -> OutDataT | Sequence[OutDataT]:
        if not self._is_open:
            raise RuntimeError("Dataset.open() was not called")

        if isinstance(index, int):
            if index < 0 or index >= self._observations:
                raise IndexError("Index out of range")
            return self._reader.read(index * self._sequence_length, self._sequence_length)

        if isinstance(index, slice):
            start, stop, step = index.indices(self._observations)
            if min(start, stop) < 0 or max(start, stop) > self._observations:
                raise IndexError("Index out of range")

            return [
                self._reader.read(i * self._sequence_length, self._sequence_length)
                for i in range(start, stop, step)
            ]
        raise NotImplementedError(f"unsupported input {index}")


@final
class StreamAsyncReaderDataset[OutDataT](DatasetAsyncReaderAbstract[OutDataT]):
    def __init__(self, storage: StreamStorageAbstract[Any, OutDataT], sequence_length: int) -> None:
        self._reader = storage.async_reader()
        self._sequence_length = sequence_length
        self._observations = 0
        self._is_open = False

    async def open(self) -> None:
        if self._is_open:
            return

        self._is_open = True
        await self._reader.open()
        self._observations = self._reader.total_length // self._sequence_length

    async def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False
        await self._reader.close()

    def __len__(self) -> int:
        if not self._is_open:
            raise RuntimeError("Dataset.open() was not called")

        return self._observations

    @overload
    def __getitem__(self, index: int) -> Awaitable[OutDataT]: ...

    @overload
    def __getitem__(self, index: slice) -> Awaitable[Sequence[OutDataT]]: ...

    def __getitem__(self, index: int | slice) -> Awaitable[OutDataT] | Awaitable[Sequence[OutDataT]]:
        if isinstance(index, int):
            if index < 0 or index >= self._observations:
                raise IndexError("Index out of range")
            return self._reader.read(index * self._sequence_length, self._sequence_length)

        if isinstance(index, slice):
            start, stop, step = index.indices(self._observations)
            if min(start, stop) < 0 or max(start, stop) > self._observations:
                raise IndexError("Index out of range")

            async def _read_many() -> Sequence[OutDataT]:
                return await asyncio.gather(
                    *(
                        self._reader.read(i * self._sequence_length, self._sequence_length)
                        for i in range(start, stop, step)
                    )
                )

            return _read_many()
        raise NotImplementedError(f"unsupported input {index}")
