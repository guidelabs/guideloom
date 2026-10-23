from typing import Protocol, runtime_checkable

import numpy as np

from .primitive import Vector


@runtime_checkable
class StreamStorageFormat[InDataT, OutDataT, StreamDType: np.generic](Protocol):
    stream_dtype: np.dtype[StreamDType]

    def __init__(self, stream_dtype: np.dtype[StreamDType]) -> None:
        """Create the storage format.

        Example:
            StreamStorageFormat(stream_dtype=np.dtype(np.int64))

        Args:
            stream_dtype (np.dtype[StreamDType]): the `numpy.dtype` of the stream data.
        """
        self.stream_dtype = stream_dtype

    def pack(self, data: InDataT) -> Vector[StreamDType]:
        """Pack the data for the database to store.

        Example:
            return data['tokens']

        Args:
            data (InDataT): user provided data structure.

        Returns:
            Vector[StreamDType]: The stream data, should be a 1d numpy vector. A structured dtype can
            be used to store more complex data, such as multiple columns.
        """
        ...

    def unpack(
        self, stream: Vector[StreamDType], shard_lengths: Vector[np.int64], shards: Vector[np.int64]
    ) -> OutDataT:
        """Decode a window from concatenated stream segments.

        `stream` is the concatenation of the arrays returned by `pack` for this read. Segment `i`
        is `shard_lengths[i]` elements long and comes from `shards[i]`. The segments are in order,
        so `stream[:shard_lengths[0]]` comes from `shards[0]` when the window is non-empty.

        Example:
            shard_ids = np.concat([
                np.full((length,), shard, dtype=np.int64)
                for length, shard in zip(shard_lengths.tolist(), shards.tolist(), strict=True)
            ])
            return {"tokens": stream, "shard": shard_ids}

        Args:
            stream (Vector[StreamDType]): Concatenated stream elements for this read.
            shard_lengths (Vector[np.int64]): Length of each contiguous shard segment, in read order.
            shards (Vector[np.int64]): Shard id of each segment, parallel to `shard_lengths`.

        Returns:
            OutDataT: Decoded window.
        """
        ...
