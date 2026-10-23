from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import pathlib

from .primitive import StorageBufferTypes


class IOReaderAbstract(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def is_supported() -> bool:
        """Detects if this I/O interface is supported.

        Returns:
            bool: true if supported.
        """
        ...

    @abstractmethod
    def sync_reader(self) -> Callable[[pathlib.PurePath], "IOSyncReaderAbstract"]:
        """Get the constructor for the sync reader.

        Returns:
            Callable[[pathlib.PurePath], IOSyncReaderAbstract]: sync reader constructor.
        """
        ...

    @abstractmethod
    def async_reader(self) -> Callable[[pathlib.PurePath], "IOAsyncReaderAbstract"]:
        """Get the constructor for the async reader.

        Returns:
            Callable[[pathlib.PurePath], IOAsyncReaderAbstract]: async reader constructor.
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
