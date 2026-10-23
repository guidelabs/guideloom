from abc import ABC, ABCMeta, abstractmethod

from .context_manager import AsyncContextManager, SyncContextManager


class StreamStorageAbstract[InDataT, OutDataT](metaclass=ABCMeta):
    @abstractmethod
    def sync_writer(self) -> "StreamStorageSyncWriterAbstract[InDataT]":
        """Return a synchronous writer that appends to explicit shards.

        Use the writer as a context manager.

        Returns:
            StreamStorageSyncWriterAbstract[InDataT]: Sync writer instance.
        """
        ...

    @abstractmethod
    def sync_reader(self) -> "StreamStorageSyncReaderAbstract[OutDataT]":
        """Return a synchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            StreamStorageSyncReaderAbstract[OutDataT]: Sync reader instance.
        """
        ...

    @abstractmethod
    def async_reader(self) -> "StreamStorageAsyncReaderAbstract[OutDataT]":
        """Return an asynchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            StreamStorageAsyncReaderAbstract[OutDataT]: Async reader instance.
        """
        ...


class StreamStorageSyncWriterAbstract[InDataT](SyncContextManager, ABC):
    @abstractmethod
    def append(self, shard: int, data: InDataT) -> None:
        """Append one encoded record to `shard`.

        The storage format's `pack` function encodes `data`. The written length is the number of
        stream elements in that record.

        Args:
            shard (int): Shard to write.
            data (InDataT): User data to encode and append.
        """
        ...

    @abstractmethod
    def length(self, shard: int) -> int:
        """Return the stream elements appended to `shard` since `open()`.

        This count is not the on-disk size of an existing shard. A shard that has not been appended
        to in this session raises `KeyError`.

        Args:
            shard (int): Shard written during this session.

        Returns:
            int: Stream elements appended to `shard` since this writer was opened.
        """
        ...


class _StreamStorageSharedReaderAbstract(metaclass=ABCMeta):
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
        """The total number of stream elements (e.g. tokens) in the database.

        Returns:
            int: number of stream elements.
        """
        ...

    @abstractmethod
    def length(self, shard: int) -> int:
        """Return the number of stream elements stored in the shard.

        `shard` must satisfy `0 <= shard < num_shards`. An out-of-range shard raises `IndexError`.

        Args:
            shard (int): Shard index.

        Returns:
            int: Stream elements in the shard.
        """
        ...


class StreamStorageSyncReaderAbstract[OutDataT](_StreamStorageSharedReaderAbstract, SyncContextManager, ABC):
    @abstractmethod
    def read(self, global_index: int, global_length: int) -> OutDataT:
        """Read `global_length` stream elements starting at `global_index`.

        `global_index` is an element offset across all shards and must lie in the dataset. An index
        past the end raises `IndexError`. A negative `global_length` raises `IndexError`.

        The read continues into the next shard when it passes the end of the current one. Past the
        last element it continues at shard 0. A `global_length` longer than the dataset wraps more
        than once.

        The result is the value returned by `storage_format.unpack`. A plain stream format receives
        the concatenated elements, the length of each shard segment, and the shard id of each
        segment. A metadata format receives the stream plus the metadata payloads for that window.

        Args:
            global_index (int): Element offset where the read starts, across all shards.
            global_length (int): Number of stream elements to read. The span may cross shards.

        Returns:
            OutDataT: Value returned by the storage format.
        """
        ...


class StreamStorageAsyncReaderAbstract[OutDataT](
    _StreamStorageSharedReaderAbstract, AsyncContextManager, ABC
):
    @abstractmethod
    async def read(self, global_index: int, global_length: int) -> OutDataT:
        """Read `global_length` stream elements starting at `global_index`.

        `global_index` is an element offset across all shards and must lie in the dataset. An index
        past the end raises `IndexError`. A negative `global_length` raises `IndexError`.

        The read continues into the next shard when it passes the end of the current one. Past the
        last element it continues at shard 0. A `global_length` longer than the dataset wraps more
        than once.

        The result is the value returned by `storage_format.unpack`. A plain stream format receives
        the concatenated elements, the length of each shard segment, and the shard id of each
        segment. A metadata format receives the stream plus the metadata payloads for that window.

        Args:
            global_index (int): Element offset where the read starts, across all shards.
            global_length (int): Number of stream elements to read. The span may cross shards.

        Returns:
            OutDataT: Value returned by the storage format.
        """
        ...
