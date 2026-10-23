from collections.abc import Generator, Hashable, Sequence

from numba import int32, int64, njit
from numba.experimental import jitclass
import numpy as np

from ..types import StorageBufferTypes, StreamWithMetaStorageFormat, UserBufferTypes, Vector
from .types import StreamedConceptData, TokenizedConceptDocument

INT32_DTYPE = np.dtype(np.int32)

_spec = [
    ("stride", int64),
    ("doc_index", int32),
    ("doc_chunk_id", int32),
    ("concept", int32[:]),
]


@jitclass(_spec)
class ChunkInfo:
    def __init__(self, stride: int, doc_index: int, doc_chunk_id: int, concept: Vector[np.int32]) -> None:
        self.stride = stride
        self.doc_index = doc_index
        self.doc_chunk_id = doc_chunk_id
        self.concept = concept


@njit(nogil=True)
def _deserialize_document(stride: int, encoded: StorageBufferTypes) -> ChunkInfo:
    data = np.frombuffer(encoded, dtype=INT32_DTYPE)
    return ChunkInfo(stride, data[0], data[1], data[2:])


class ConceptStorageFormat(
    StreamWithMetaStorageFormat[TokenizedConceptDocument, StreamedConceptData, np.int32]
):
    def __init__(self) -> None:
        super().__init__(INT32_DTYPE)
        self._concept_dtype = INT32_DTYPE.newbyteorder("<")
        self._concept_dtype_be = INT32_DTYPE.newbyteorder(">")

    def size(self, data: TokenizedConceptDocument) -> int:
        return data["input_ids"].size

    def pack(
        self, data: TokenizedConceptDocument
    ) -> Generator[tuple[Hashable, UserBufferTypes | None], None, Vector[np.int32]]:
        concepts = data["concepts"]
        prev_chunk_id: int | None = None

        for raw_chunk_id in data["chunk_ids"]:
            chunk_id = int(raw_chunk_id)
            if prev_chunk_id != chunk_id:
                chunk_encoded = self.serialize(data["doc_index"], chunk_id, concepts[chunk_id])
                yield ((chunk_id,), chunk_encoded)
            else:
                yield ((chunk_id,), None)
            prev_chunk_id = chunk_id

        return data["input_ids"]

    def unpack(
        self,
        stream_data: Vector[np.int32],
        document_shard: Vector[np.int64],
        document_index: Vector[np.int32],
        document_data: Sequence[StorageBufferTypes],
    ) -> StreamedConceptData:
        return {
            "chunks": [
                _deserialize_document(shard, encoded)
                for shard, encoded in zip(document_shard.tolist(), document_data, strict=True)
            ],
            "chunk_ids": document_index,
            "input_ids": stream_data,
        }

    def serialize(self, doc_index: int, doc_chunk_id: int, concept: Vector[np.int32]) -> bytes:
        if concept.dtype == self._concept_dtype_be:
            concept = concept.view(self._concept_dtype).byteswap()
        if concept.dtype != self._concept_dtype:
            raise ValueError(f"concept dtype, expected ({self._concept_dtype}) was ({concept.dtype})")
        if concept.ndim != 1:
            raise ValueError("value numpy array must be 1-D")

        prefix = np.array([doc_index, doc_chunk_id], dtype=self._concept_dtype)
        return prefix.tobytes() + concept.tobytes(order="C")
