# Changelog

## Unreleased

- `StatefulDataLoader` requires `shuffle_seed` when `shuffle=True` and raises `ValueError`
  otherwise. Previously, with the default `shuffle_seed=None`, each rank drew its own
  permutation, so ranks read overlapping samples and skipped others. Pass the same integer seed
  on every rank.
- The `load_state_dict` warning for a mismatched `shuffle_seed` now shows the loader's seed
  instead of its `shuffle` flag.

## 1.1.0

Improvements to the type defintions for the paths. Mostly helpful when implementing
custom IO backends that takes special PurePath types.

- `IOReaderAbstract` is generic over its path type (`PathT: pathlib.PurePath = pathlib.PurePath`),
  and that path type propagates to `DocumentStorage`, `StreamStorage`, `StreamWithMetaStorage`,
  `FileSystemDocument`, and `FileSystemStream`.
- `SeekIO`, `MmapIO`, and `PosixPreadIO` require `pathlib.Path`, as does omitting `read_interface`.

## 1.0.0

This is the first public release. Compared to prior internal releases,
the following changes has been made.

- Documentation, typed-package metadata, and automated release validation.
- Import the main classes from `guideloom` and extension types from `guideloom.types`.
- Loader checkpoints store `observation_offset` instead of `batch_step`, preserving global sample
  order when resuming with a different batch size or world size.
- The asynchronous dataset now returns a single awaitable.
- An `read_many` method has been added to the storage backedn.
- Safe repeated POSIX reader cleanup. Deterministic shuffling uses a GIL-releasing Numba Feistel permutation.
- PyTorch and torchdata are optional. Install the `torch` extra for `StatefulDataLoader`.
