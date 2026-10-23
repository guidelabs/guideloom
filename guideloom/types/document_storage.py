from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
import asyncio

import numpy as np

from .context_manager import AsyncContextManager, SyncContextManager
from .primitive import Vector


class DocumentStorageAbstract[InDataT, OutDataT](metaclass=ABCMeta):
    @abstractmethod
    def sync_writer(self) -> "DocumentStorageSyncWriterAbstract[InDataT]":
        """Return a synchronous writer that appends to explicit shards.

        Use the writer as a context manager.

        Returns:
            DocumentStorageSyncWriterAbstract[InDataT]: Sync writer instance.
        """
        ...

    @abstractmethod
    def sync_reader(self) -> "DocumentStorageSyncReaderAbstract[OutDataT]":
        """Return a synchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            DocumentStorageSyncReaderAbstract[OutDataT]: Sync reader instance.
        """
        ...

    @abstractmethod
    def async_reader(self) -> "DocumentStorageAsyncReaderAbstract[OutDataT]":
        """Return an asynchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            DocumentStorageAsyncReaderAbstract[OutDataT]: Async reader instance.
        """
        ...


class DocumentStorageSyncWriterAbstract[InDataT](SyncContextManager, ABC):
    @abstractmethod
    def append(self, shard: int, data: InDataT) -> None:
        """Append one encoded document to `shard`.

        The storage format's `pack` function encodes `data`. Built-in storage accepts `bytes` or
        `memoryview` from `pack`.

        Args:
            shard (int): Shard to write.
            data (InDataT): User data to encode and append.
        """
        ...

    @abstractmethod
    def length(self, shard: int) -> int:
        """Return the documents appended to `shard` since `open()`.

        This count is not the on-disk size of an existing shard. A shard that has not been appended
        to in this session raises `KeyError`.

        Args:
            shard (int): Shard written during this session.

        Returns:
            int: Documents appended to `shard` since this writer was opened.
        """
        ...


class _DocumentStorageSharedReaderAbstract(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_shards(self) -> int:
        """The total number of shards in the database.

        Returns:
            int: number of shards.
        """
        ...

    @property
    @abstractmethod
    def total_length(self) -> int:
        """The total number of documents in the database.

        Returns:
            int: number of documents.
        """
        ...

    @abstractmethod
    def length(self, shard: int) -> int:
        """Return the number of documents stored in the shard.

        `shard` must satisfy `0 <= shard < num_shards`. An out-of-range shard raises `IndexError`.

        Args:
            shard (int): Shard index.

        Returns:
            int: Documents in the shard.
        """
        ...


class DocumentStorageSyncReaderAbstract[OutDataT](
    _DocumentStorageSharedReaderAbstract, SyncContextManager, ABC
):
    @abstractmethod
    def read(self, global_index: int) -> OutDataT:
        """Read the document at `global_index`.

        `global_index` is a document offset across all shards. An index outside the dataset raises
        `IndexError`.

        Args:
            global_index (int): Document offset across all shards.

        Returns:
            OutDataT: Value returned by `storage_format.unpack`.
        """
        ...

    def read_many(self, global_indexes: Vector[np.int64]) -> Sequence[OutDataT]:
        """Read the documents at `global_indexes`.

        Overrides may read more efficiently than one `read` call per index. The result matches
        `[self.read(int(i)) for i in global_indexes]`, including order.

        Args:
            global_indexes (Vector[np.int64]): Document offsets across all shards.

        Returns:
            Sequence[OutDataT]: Values returned by `storage_format.unpack`, in request order.
        """
        return [self.read(int(global_index)) for global_index in global_indexes]


class DocumentStorageAsyncReaderAbstract[OutDataT](
    _DocumentStorageSharedReaderAbstract, AsyncContextManager, ABC
):
    @abstractmethod
    async def read(self, global_index: int) -> OutDataT:
        """Read the document at `global_index`.

        `global_index` is a document offset across all shards. An index outside the dataset raises
        `IndexError`.

        Args:
            global_index (int): Document offset across all shards.

        Returns:
            OutDataT: Value returned by `storage_format.unpack`.
        """
        ...

    async def read_many(self, global_indexes: Vector[np.int64]) -> Sequence[OutDataT]:
        """Read the documents at `global_indexes`.

        Overrides may read more efficiently than one `read` call per index. The result matches
        `await asyncio.gather(*(self.read(int(i)) for i in global_indexes))`, including order.

        Args:
            global_indexes (Vector[np.int64]): Document offsets across all shards.

        Returns:
            Sequence[OutDataT]: Values returned by `storage_format.unpack`, in request order.
        """
        return await asyncio.gather(*(self.read(int(global_index)) for global_index in global_indexes))
