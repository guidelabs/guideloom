"""Stateless Feistel index shuffle.

Shufflers with the same domain size, round count, and seed reuse two scalar Numba
kernels through a process-local LRU cache. The kernels release the GIL. There is
no Numba disk cache or vectorized API. The gfc-derived implementation and its
attribution are in `guideloom.shuffle._gfc`.
"""

from typing import Self

import numpy as np

from ._gfc import specialized_scalar_kernels


class FeistelShuffler:
    __slots__ = ("_forward", "_inverse", "length", "rounds")

    def __init__(self, length: int, rng: np.random.Generator, rounds: int = 5) -> None:
        """Build a stateless permutation of indexes `0 <= i < length`.

        Args:
            length (int): Domain size N.
            rng (np.random.Generator): Generator used to seed the Feistel network.
            rounds (int): Feistel rounds. 4 to 6 is enough for shuffling. Defaults to 5.
        """
        self.length = length
        self.rounds = rounds

        full_seed = int(rng.integers(low=0, high=2**32, dtype=np.uint64))
        self._forward, self._inverse = specialized_scalar_kernels(length, rounds, full_seed)

    @classmethod
    def from_seed(cls, length: int, seed: int | None = None, rounds: int = 5) -> Self:
        """Build a shuffler from an integer seed.

        Args:
            length (int): Domain size N.
            seed (int | None): Seed for the round keys. `None` draws nondeterministic entropy.
                Defaults to None.
            rounds (int): Feistel rounds. 4 to 6 is enough for shuffling. Defaults to 5.

        Returns:
            Self: Shuffler initialized from `seed`.
        """
        rng = np.random.default_rng(seed)
        return cls(length, rng, rounds)

    def shuffle(self, i: int) -> int:
        """Map an index to its shuffled position.

        The map is a permutation of `0 <= i < self.length`. An index outside that domain raises
        `IndexError`.

        Args:
            i (int): Index in that half-open range.

        Returns:
            int: Shuffled index.
        """
        return int(self._forward(np.uint64(i)))

    def unshuffle(self, y: int) -> int:
        """Map a shuffled index back to its original position.

        This is the inverse of `shuffle`. An index outside `0 <= y < self.length` raises `IndexError`.

        Args:
            y (int): Shuffled index in that half-open range.

        Returns:
            int: Original index.
        """
        return int(self._inverse(np.uint64(y)))
