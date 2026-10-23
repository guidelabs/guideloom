from typing import Protocol, runtime_checkable

from .primitive import StorageBufferTypes, UserBufferTypes


@runtime_checkable
class DocumentStorageFormat[InDataT, OutDataT](Protocol):
    def pack(self, data: InDataT) -> UserBufferTypes:
        """Encode one document for storage.

        Built-in document storage accepts `bytes` or `memoryview` from this method.

        Example:
            return pickle.dumps(data)

        Args:
            data (InDataT): User-provided document.

        Returns:
            UserBufferTypes: Encoded document. Built-in storage accepts `bytes` or `memoryview`.
        """
        ...

    def unpack(self, encoded_data: StorageBufferTypes, shard: int) -> OutDataT:
        """Decode one document.

        Storage does not keep a document index. Encode one in `pack` when you need it. Together with
        `shard`, that identifies the shard and the document within that shard.

        Example:
            return pickle.loads(encoded_data)

        Args:
            encoded_data (StorageBufferTypes): Bytes produced by `pack`.
            shard (int): Shard the document was read from.

        Returns:
            OutDataT: Decoded document.
        """
        ...
