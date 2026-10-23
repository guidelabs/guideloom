from collections.abc import Awaitable, Sequence
from typing import Any, final, overload

import numpy as np

from ..types import (
    DatasetAbstract,
    DatasetAsyncReaderAbstract,
    DatasetSyncReaderAbstract,
    DocumentStorageAbstract,
)


@final
class DocumentDataset[OutDataT](DatasetAbstract[OutDataT]):
    def __init__(self, storage: DocumentStorageAbstract[Any, OutDataT]) -> None:
        """Create a dataset view of document storage, with one item per document.

        The opened view length is the number of documents. Open a reader before indexing it.

        Args:
            storage (DocumentStorageAbstract[Any, OutDataT]): Storage that provides access to the database.
        """
        self._storage = storage

    def sync_reader(self) -> "DocumentSyncReaderDataset[OutDataT]":
        return DocumentSyncReaderDataset(self._storage)

    def async_reader(self) -> "DocumentAsyncReaderDataset[OutDataT]":
        return DocumentAsyncReaderDataset(self._storage)


@final
class DocumentSyncReaderDataset[OutDataT](DatasetSyncReaderAbstract[OutDataT]):
    def __init__(self, storage: DocumentStorageAbstract[Any, OutDataT]) -> None:
        self._reader = storage.sync_reader()
        self._observations = 0
        self._is_open = False

        self.open()

    def open(self) -> None:
        if self._is_open:
            return

        self._is_open = True
        self._reader.open()
        self._observations = self._reader.total_length

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
        if isinstance(index, int):
            if index < 0 or index >= self._observations:
                raise IndexError("Index out of range")
            return self._reader.read(index)

        if isinstance(index, slice):
            start, stop, step = index.indices(self._observations)
            if min(start, stop) < 0 or max(start, stop) > self._observations:
                raise IndexError("Index out of range")

            return self._reader.read_many(np.arange(start, stop, step, dtype=np.int64))
        raise NotImplementedError(f"unsupported input {index}")


@final
class DocumentAsyncReaderDataset[OutDataT](DatasetAsyncReaderAbstract[OutDataT]):
    def __init__(self, storage: DocumentStorageAbstract[Any, OutDataT]) -> None:
        self._reader = storage.async_reader()
        self._observations = 0
        self._is_open = False

    async def open(self) -> None:
        if self._is_open:
            return

        self._is_open = True
        await self._reader.open()
        self._observations = self._reader.total_length

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
            return self._reader.read(index)

        if isinstance(index, slice):
            start, stop, step = index.indices(self._observations)
            if min(start, stop) < 0 or max(start, stop) > self._observations:
                raise IndexError("Index out of range")

            return self._reader.read_many(np.arange(start, stop, step, dtype=np.int64))
        raise NotImplementedError(f"unsupported input {index}")
