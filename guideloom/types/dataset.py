from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Awaitable, Sequence
from typing import Any, overload

from .context_manager import AsyncContextManager, SyncContextManager
from .document_storage import DocumentStorageAbstract
from .stream_storage import StreamStorageAbstract


class DatasetAbstract[OutT](metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self, storage: DocumentStorageAbstract[Any, OutT] | StreamStorageAbstract[Any, OutT]
    ) -> None: ...

    @abstractmethod
    def sync_reader(self) -> "DatasetSyncReaderAbstract[OutT]":
        """Return a synchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            DatasetSyncReaderAbstract[OutT]: Sync reader instance.
        """
        ...

    @abstractmethod
    def async_reader(self) -> "DatasetAsyncReaderAbstract[OutT]":
        """Return an asynchronous reader over the entire dataset.

        Use the reader as a context manager.

        Returns:
            DatasetAsyncReaderAbstract[OutT]: Async reader instance.
        """
        ...


class DatasetSyncReaderAbstract[OutT](Sequence[OutT], SyncContextManager, ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of observations in the opened dataset view.

        Calling this before the reader is opened raises `RuntimeError`.

        Returns:
            int: Number of observations.
        """
        ...

    @overload
    def __getitem__(self, index: int) -> OutT: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[OutT]: ...

    @abstractmethod
    def __getitem__(self, index: int | slice) -> OutT | Sequence[OutT]:
        """Return the observation at `index`.

        An integer index must satisfy `0 <= index < len(self)`. A negative integer raises
        `IndexError`. A slice uses normal slice semantics over the opened view.

        Args:
            index (int | slice): Integer position or slice within the opened view.

        Returns:
            OutT | Sequence[OutT]: One observation, or a sequence of observations for a slice.
        """
        ...


class DatasetAsyncReaderAbstract[OutT](AsyncContextManager, ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of observations in the opened dataset view.

        Calling this before the reader is opened raises `RuntimeError`.

        Returns:
            int: Number of observations.
        """
        ...

    @overload
    def __getitem__(self, index: int) -> Awaitable[OutT]: ...

    @overload
    def __getitem__(self, index: slice) -> Awaitable[Sequence[OutT]]: ...

    @abstractmethod
    def __getitem__(self, index: int | slice) -> Awaitable[OutT] | Awaitable[Sequence[OutT]]:
        """Return an awaitable observation at `index`.

        An integer index must satisfy `0 <= index < len(self)`. A negative integer raises
        `IndexError`. A slice uses normal slice semantics over the opened view.

        Args:
            index (int | slice): Integer position or slice within the opened view.

        Returns:
            Awaitable[OutT] | Awaitable[Sequence[OutT]]: Awaitable observation, or an awaitable
                sequence of observations for a slice.
        """
        ...
