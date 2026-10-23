from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload
import itertools

import numpy as np

from ..types import Vector
from .types import TokenizedConceptDocument


@dataclass(eq=False, frozen=True)
class FakeConceptData(Sequence[TokenizedConceptDocument]):
    seed: int | None = None
    stride: int = 0
    observations: int = 100
    max_chunks_per_observation: int = 50
    max_concepts_per_chunk: int = 40
    max_tokens_per_chunk: int = 200
    num_of_concepts: int = 100_000
    vocab: int = 100_000

    def _get_observation(self, index: int) -> TokenizedConceptDocument:
        bit_gen = np.random.Philox(np.random.SeedSequence(self.seed, spawn_key=(self.stride, index)))
        rng = np.random.Generator(bit_gen)

        num_chunks = int(rng.integers(1, self.max_chunks_per_observation, endpoint=True, dtype=np.int32))
        chunk_lens = rng.integers(
            1, self.max_tokens_per_chunk, size=(num_chunks,), endpoint=True, dtype=np.int32
        )
        input_ids = rng.integers(0, self.vocab, size=int(chunk_lens.sum()), dtype=np.int32)
        chunk_ids = np.repeat(np.arange(num_chunks, dtype=np.int32), chunk_lens)

        concept_lens = rng.integers(
            0, self.max_concepts_per_chunk, size=(num_chunks,), endpoint=True, dtype=np.int32
        )
        concept_boundary = np.concatenate(([0], np.cumsum(concept_lens)), axis=0)
        concept_idx = rng.integers(
            0, self.num_of_concepts, size=int(concept_boundary[-1]), endpoint=False, dtype=np.int32
        )
        concepts: list[Vector[np.int32]] = [
            concept_idx[start:end] for start, end in itertools.pairwise(concept_boundary)
        ]

        return {
            "doc_index": index,
            "doc_id": f"fake_{self.stride}_{index}".encode(),
            "concepts": concepts,
            "chunk_ids": chunk_ids,
            "input_ids": input_ids,
        }

    def __len__(self) -> int:
        return self.observations

    @overload
    def __getitem__(self, index: int) -> TokenizedConceptDocument: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TokenizedConceptDocument]: ...

    def __getitem__(
        self, index: slice | int
    ) -> TokenizedConceptDocument | Sequence[TokenizedConceptDocument]:
        if isinstance(index, int):
            if index < 0 or index >= self.observations:
                raise IndexError("Index out of range")
            return self._get_observation(index)

        if isinstance(index, slice):
            start, stop, step = index.indices(self.observations)
            return [self._get_observation(i) for i in range(start, stop, step)]

        raise NotImplementedError(f"unsupported input {index}")
