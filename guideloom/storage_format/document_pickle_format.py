import pickle

from ..types import DocumentStorageFormat, StorageBufferTypes


class DocumentPickleFormat[DocumentT](DocumentStorageFormat[DocumentT, DocumentT]):
    """Pickle each document with `pickle.dumps` and `pickle.loads`.

    `unpack` ignores `shard`.
    """

    def pack(self, data: DocumentT) -> bytes:
        """Pickle a document.

        Args:
            data (DocumentT): Document to store.

        Returns:
            bytes: Pickled document.
        """
        return pickle.dumps(data)

    def unpack(self, encoded_data: StorageBufferTypes, shard: int) -> DocumentT:
        """Unpickle a document.

        Args:
            encoded_data (StorageBufferTypes): Bytes produced by `pack`.
            shard (int): Shard the document was read from. Ignored.

        Returns:
            DocumentT: Unpickled document.
        """
        return pickle.loads(encoded_data)
