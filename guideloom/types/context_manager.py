from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Self


class SyncContextManager(AbstractContextManager["SyncContextManager", None], ABC):
    def __enter__(self) -> Self:
        super().__enter__()
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Open this context."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close this context."""
        ...


class AsyncContextManager(AbstractAsyncContextManager["AsyncContextManager", None], ABC):
    async def __aenter__(self) -> Self:
        await super().__aenter__()
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @abstractmethod
    async def open(self) -> None:
        """Open this context."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close this context."""
        ...
