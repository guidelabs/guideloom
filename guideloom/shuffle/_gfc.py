# Copyright (c) 2021 Maxime Mouchet
#
# Adapted from gfc under the MIT license.
# See licenses/gfc.txt in the source distribution
# (or the installed distribution's .dist-info/licenses directory).
# Source: https://github.com/junekiwi/gfc/blob/main/src/gfc.c
# gfc's Speck code credits madmo/speck (ISC): https://github.com/madmo/speck

from collections.abc import Callable
from functools import lru_cache
import math

from numba import njit, uint64
import numpy as np

type _KeySchedule = np.ndarray[tuple[int, int], np.dtype[np.uint32]]


@njit(nogil=True)
def _make_round_keys(rounds: int, seed: np.uint64) -> _KeySchedule:
    # Expand the Speck64/128 key schedule to (rounds, 27) uint32 words.
    keys = np.empty((rounds, 27), dtype=np.uint32)
    for j in range(rounds):
        # C casts uint64[2] {seed, j} to uint32[4] (little endian).
        b = np.uint32(seed)
        a = np.empty(3, dtype=np.uint32)
        a[0] = np.uint32(seed >> np.uint64(32))
        a[1] = np.uint32(j)
        a[2] = np.uint32(np.uint64(j) >> np.uint64(32))
        keys[j, 0] = b
        for k in range(26):
            idx = k % 3
            x = a[idx]
            x = np.uint32(((x >> 8) | (x << 24)) + b)
            x = np.uint32(x ^ np.uint32(k))
            b = np.uint32(((b << 3) | (b >> 29)) ^ x)
            a[idx] = x
            keys[j, k + 1] = b
    return keys


@njit(nogil=True, error_model="numpy", inline="always")
def _speck64(block: np.uint64, round_index: int, keys: _KeySchedule) -> np.uint64:
    # Speck64/128 block encryption, with its 27 steps grouped in threes.
    # The cipher itself uses 32-bit words; the combined result is 64-bit.
    # Grouping 3 steps reduces loop overhead without explicitly writing out
    # all 27 operations, keeping compilation cost manageable.
    lo = np.uint32(block)
    hi = np.uint32(block >> np.uint64(32))
    for group in range(9):
        k = 3 * group
        hi = np.uint32(((hi >> 8) | (hi << 24)) + lo) ^ keys[round_index, k]
        lo = np.uint32(((lo << 3) | (lo >> 29)) ^ hi)

        hi = np.uint32(((hi >> 8) | (hi << 24)) + lo) ^ keys[round_index, k + 1]
        lo = np.uint32(((lo << 3) | (lo >> 29)) ^ hi)

        hi = np.uint32(((hi >> 8) | (hi << 24)) + lo) ^ keys[round_index, k + 2]
        lo = np.uint32(((lo << 3) | (lo >> 29)) ^ hi)
    return (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)


@njit(nogil=True, error_model="numpy", inline="always")
def _feistel(value: np.uint64, a: np.uint64, keys: _KeySchedule) -> np.uint64:
    left = value % a
    right = value // a
    for j in range(keys.shape[0]):
        new_right = (left + _speck64(right, j, keys)) % a
        left = right
        right = new_right
    if keys.shape[0] & 1:
        return a * left + right
    return a * right + left


@njit(nogil=True, error_model="numpy", inline="always")
def _feistel_inverse(value: np.uint64, a: np.uint64, keys: _KeySchedule) -> np.uint64:
    if keys.shape[0] & 1:
        right = value % a
        left = value // a
    else:
        left = value % a
        right = value // a

    for j in range(keys.shape[0] - 1, -1, -1):
        step = _speck64(left, j, keys) % a
        new_left = (right + a - step) % a
        right = left
        left = new_left
    return a * right + left


@lru_cache(maxsize=128)
def specialized_scalar_kernels(
    size: int, rounds: int, seed: int
) -> tuple[Callable[[np.uint64], np.uint64], Callable[[np.uint64], np.uint64]]:
    """Return cached scalar forward and inverse permutations for a domain and seed.

    A process-local LRU cache retains kernels for the 128 most recently used combinations
    of size, rounds, and seed. Cache misses compile Numba kernels with explicit uint64 input
    and output signatures. They capture the domain dimensions and immutable expanded Speck
    keys as constants and release the GIL during execution. There is no Numba disk cache.

    Domain setup and cycle walking follow gfc; JIT specialization, kernel caching, immutable
    keys, and bounds checks are GuideLoom additions. Cycle walking restricts the Feistel
    permutation to `0 <= index < size` while preserving invertibility.

    Args:
        size (int): Number of indices in the domain. Zero produces kernels that reject every index.
        rounds (int): Number of Feistel rounds, each with its own expanded Speck key.
        seed (int): Unsigned 64-bit integer used to derive the round keys.

    Returns:
        tuple[Callable[[np.uint64], np.uint64], Callable[[np.uint64], np.uint64]]:
            Forward and inverse kernels, in that order. Each accepts and returns a scalar
            uint64 index and raises `IndexError` for an index greater than or equal to `size`.
            For every valid index, applying the inverse to the forward result recovers the index.
    """
    side = math.isqrt(size - 1) + 1 if size else 0
    domain_size = np.uint64(size)
    domain_side = np.uint64(side)
    keys = _make_round_keys(rounds, np.uint64(seed))
    keys.flags.writeable = False  # Let Numba treat the array as constant.

    # Explicit uint64 signature preserves high unsigned 64-bit Python ints.
    # Reject out-of-domain indices before Feistel or modulo (including size=0).
    @njit(uint64(uint64), nogil=True)
    def forward(index: np.uint64) -> np.uint64:
        if index >= domain_size:
            raise IndexError(index)
        value = _feistel(index, domain_side, keys)
        while value >= domain_size:
            value = _feistel(value, domain_side, keys)
        return value

    @njit(uint64(uint64), nogil=True)
    def inverse(index: np.uint64) -> np.uint64:
        if index >= domain_size:
            raise IndexError(index)
        value = _feistel_inverse(index, domain_side, keys)
        while value >= domain_size:
            value = _feistel_inverse(value, domain_side, keys)
        return value

    return forward, inverse
