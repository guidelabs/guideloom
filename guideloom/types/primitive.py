import numpy as np

type UserBufferTypes = bytearray | bytes | memoryview
type StorageBufferTypes = bytearray | memoryview
type Vector[DType: np.generic] = np.ndarray[tuple[int], np.dtype[DType]]
