from collections.abc import Mapping
from typing import cast

import numpy as np

from ..types import StreamStorageFormat, Vector


class StreamNumpyFormat[DataT: Mapping[str, object], DType: np.void](
    StreamStorageFormat[DataT, DataT, DType]
):
    """Pack a mapping of equal-length arrays into one structured stream array.

    `unpack` returns those fields and ignores shard segmentation.
    """

    def __init__(self, stream_dtype: np.dtype[DType], data_type: type[DataT] | None = None) -> None:
        """Create a format for a structured dtype.

        Args:
            stream_dtype (np.dtype[DType]): Structured dtype. Its field names are the mapping keys.
            data_type (type[DataT] | None): Mapping type used only to set the `DataT` type parameter.
                Ignored at runtime. Defaults to None.

        Raises:
            ValueError: If `stream_dtype` is not a structured dtype.
        """
        if stream_dtype.names is None:
            raise ValueError("dtype must be a structured dtyped")
        super().__init__(stream_dtype)

    def pack(self, data: DataT) -> Vector[DType]:
        """Stack mapping fields into one structured array.

        Every field named by `stream_dtype` must be a one-dimensional array of the same length.

        Args:
            data (DataT): Mapping from field name to array.

        Returns:
            Vector[DType]: Structured array with `stream_dtype`.
        """
        assert self.stream_dtype.names is not None
        first_arr = data[self.stream_dtype.names[0]]
        assert isinstance(first_arr, np.ndarray)
        container = np.empty((first_arr.size,), dtype=self.stream_dtype)
        for name in self.stream_dtype.names:
            container[name] = data[name]
        return container

    def unpack(
        self, stream: Vector[DType], shard_lengths: Vector[np.int64], shards: Vector[np.int64]
    ) -> DataT:
        """Return each structured field as a mapping entry.

        Args:
            stream (Vector[DType]): Structured array read from storage.
            shard_lengths (Vector[np.int64]): Length of each shard segment. Ignored.
            shards (Vector[np.int64]): Shard id of each segment. Ignored.

        Returns:
            DataT: Mapping from field name to that field's array.
        """
        assert self.stream_dtype.names is not None
        return cast("DataT", {name: stream[name] for name in self.stream_dtype.names})
