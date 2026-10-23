from collections.abc import Sequence
from typing import Protocol, TypedDict

import numpy as np

from ..types import Vector


class TokenizedConceptDocument(TypedDict):
    doc_index: int
    doc_id: bytes
    concepts: Sequence[Vector[np.int32]]
    chunk_ids: Vector[np.int32]
    input_ids: Vector[np.int32]


class ConceptItem(Protocol):
    stride: int
    doc_index: int
    doc_chunk_id: int
    concept: Vector[np.int32]


class StreamedConceptData(TypedDict):
    chunks: Sequence[ConceptItem]
    chunk_ids: Vector[np.int32]
    input_ids: Vector[np.int32]
