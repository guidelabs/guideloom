from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import pathlib

from .primitive import StorageBufferTypes


class IOReaderAbstract[PathT: pathlib.PurePath = pathlib.PurePath](metaclass=ABCMeta):
    """Factory for readers of one file.

    `PathT` is the path type passed to each file reader. It defaults to `pathlib.PurePath`. Built-in
    backends use `pathlib.Path` because they open local files. A custom backend can keep the default
    or require a `PurePath` subclass.
    """

    @staticmethod
    @abstractmethod
    def is_supported() -> bool:
        """Detects if this I/O interface is supported.

        Returns:
            bool: true if supported.
        """
        ...

    @abstractmethod
    def sync_reader(self) -> Callable[[PathT], "IOSyncReaderAbstract"]:
        """Get the constructor for the sync reader.

        Returns:
            Callable[[PathT], IOSyncReaderAbstract]: Sync reader constructor. Called with a path of
                type `PathT`.
        """
        ...

    @abstractmethod
    def async_reader(self) -> Callable[[PathT], "IOAsyncReaderAbstract"]:
        """Get the constructor for the async reader.

        Returns:
            Callable[[PathT], IOAsyncReaderAbstract]: Async reader constructor. Called with a path of
                type `PathT`.
        """
        ...


class IOSyncReaderAbstract(metaclass=ABCMeta):
    """Synchronous reader for one file.

    Implementations are constructed with the file path: ``Reader(path)``.
    """

    @abstractmethod
    def filesize(self) -> int:
        """Return the size of the file in bytes.

        Returns:
            int: Number of bytes.
        """
        ...

    @abstractmethod
    def read(self, offset: int, size: int) -> StorageBufferTypes:
        """Read `size` bytes starting at `offset`.

        Args:
            offset (int): Byte index to start reading from. A negative offset raises `IndexError`.
            size (int): Number of bytes to read. A short read raises `IndexError`.

        Returns:
            StorageBufferTypes: Bytes read, as a `bytearray` or `memoryview`.
        """
        ...

    @abstractmethod
    def open(self) -> None:
        """Open the file.

        Raises:
            FileNotFoundError: if the path cannot be found.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the file."""
        ...

    def __del__(self) -> None:
        self.close()


class IOAsyncReaderAbstract(metaclass=ABCMeta):
    """Asynchronous reader for one file.

    Implementations are constructed with the file path: ``Reader(path)``.
    """

    @abstractmethod
    async def filesize(self) -> int:
        """Return the size of the file in bytes.

        Returns:
            int: Number of bytes.
        """
        ...

    @abstractmethod
    async def read(self, offset: int, size: int) -> StorageBufferTypes:
        """Read `size` bytes starting at `offset`.

        Args:
            offset (int): Byte index to start reading from. A negative offset raises `IndexError`.
            size (int): Number of bytes to read. A short read raises `IndexError`.

        Returns:
            StorageBufferTypes: Bytes read, as a `bytearray` or `memoryview`.
        """
        ...

    @abstractmethod
    async def open(self) -> None:
        """Open the file.

        Raises:
            FileNotFoundError: if the path cannot be found.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the file."""
        ...

    @abstractmethod
    def close_sync(self) -> None:
        """Close the file synchronously.

        `__del__` calls this.
        """
        ...

    def __del__(self) -> None:
        self.close_sync()
