# Changelog

## 0.1.0 — unreleased

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
