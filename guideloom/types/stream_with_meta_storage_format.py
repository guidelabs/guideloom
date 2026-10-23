from collections.abc import Generator, Hashable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .primitive import StorageBufferTypes, UserBufferTypes, Vector


@runtime_checkable
class StreamWithMetaStorageFormat[InDataT, OutDataT, StreamDType: np.generic](Protocol):
    stream_dtype: np.dtype[StreamDType]

    def __init__(self, stream_dtype: np.dtype[StreamDType]) -> None:
        """Create the storage format.

        Example:
            StreamWithMetaStorageFormat(stream_dtype=np.dtype(np.int64))

        Args:
            stream_dtype (np.dtype[StreamDType]): the `numpy.dtype` of the stream data.
        """
        self.stream_dtype = stream_dtype

    def size(self, data: InDataT) -> int:
        """Return the number of stream elements in `data`.

        `pack` must yield this many pairs.

        Args:
            data (InDataT): Record passed to `pack`.

        Returns:
            int: Number of stream elements.
        """
        ...

    def pack(
        self, data: InDataT
    ) -> Generator[tuple[Hashable, UserBufferTypes | None], None, Vector[StreamDType]]:
        """Encode one record as one metadata pair per stream element and a stream array.

        Yield exactly `size(data)` pairs, then return a one-dimensional array with dtype
        `stream_dtype` and that same length. Return the array from the generator. Do not yield it.
        A structured dtype can store several columns in that array.

        A key may repeat only inside a contiguous run: `[0, 0, 1, 1]` is valid and `[0, 1, 0]` is
        not. The first pair in a run must include a `bytes` or `memoryview` payload. Later pairs in
        that run may pass `None` instead of repeating the payload. Keys are compared only within
        this call and are not stored. Encode document and chunk ids in the payload when readers
        need them. A key can be any hashable object.

        Example:
            ```python
            for chunk_id in data["chunk_ids"]:
                payload = data["concepts"][int(chunk_id)].tobytes()
                yield ((data["document_id"], int(chunk_id)), payload)
            return np.asarray(data["tokens"], dtype=self.stream_dtype)
            ```

        `data["chunk_ids"]` has one id per stream element, in stream order.

        Args:
            data (InDataT): User-provided record.

        Returns:
            Vector[StreamDType]: Stream array of length `size(data)`.

        Yields:
            tuple[Hashable, bytes | memoryview | None]: Key and payload for one stream element.
        """
        ...

    def unpack(
        self,
        stream_data: Vector[StreamDType],
        document_shard: Vector[np.int64],
        document_index: Vector[np.int32],
        document_data: Sequence[StorageBufferTypes],
    ) -> OutDataT:
        """Decode a window into stream elements and the metadata payloads they reference.

        `document_index` has one entry per stream element and indexes `document_data`.
        `document_shard[i]` is the shard that stored `document_data[i]`. Both are remapped for this
        read. They are not original chunk ids. Decode ids from the payload itself when you need
        provenance. The shard of the element at `token_idx` is
        `document_shard[document_index[token_idx]]`. Each payload still belongs to a single shard:
        one `append` call writes its metadata into the shard it was given, while a read can span
        several shards.

        Example:
            return {
                "tokens": stream_data,
                "annotation_index": document_index,
                "annotations": [np.frombuffer(payload, dtype=np.int32) for payload in document_data],
            }

        Args:
            stream_data (Vector[StreamDType]): Concatenated stream array for this read. It may span
                several records, so it is not aligned with one `pack` call.
            document_shard (Vector[np.int64]): Shard of each payload in `document_data`.
            document_index (Vector[np.int32]): Index into `document_data` for each stream element.
                `document_index.shape == stream_data.shape`. The payload for element `token_idx` is
                `document_data[document_index[token_idx]]`.
            document_data (Sequence[StorageBufferTypes]): Payloads for the distinct metadata objects
                referenced by this read. Decode them in `unpack`.

        Returns:
            OutDataT: Decoded window.
        """
        ...
