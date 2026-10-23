# API reference

Use these pages for constructor arguments, method signatures, and reader contracts.
For a complete preprocessing and training workflow, start with the
[quick-start](../quick-start.md).

| Module | Reference | Purpose |
| --- | --- | --- |
| `guideloom.storage` | [Storage](storage.md) | Store documents, streams, and streams with metadata; open readers and writers. |
| `guideloom.storage_format` | [Built-in storage formats](formats.md#built-in-formats) | Encode array mappings with `StreamNumpyFormat` or documents with `DocumentPickleFormat`. |
| `guideloom.io` | [Built-in I/O backends](io.md) | Choose seek, memory-mapped, or POSIX positional reads. |
| `guideloom.dataset` | [Datasets](datasets.md) | Turn stored data into indexed training samples. |
| `guideloom.pytorch` | [PyTorch DataLoader](pytorch.md) | Batch, shuffle, prefetch, partition by rank, and checkpoint sample delivery. |
| `guideloom.types` | [Format interfaces](formats.md#custom-format-interfaces) and [extension interfaces](interfaces.md) | Implement custom codecs, storage, datasets, or I/O backends. |

Storage classes, dataset classes, and built-in formats can also be imported
directly from `guideloom`. `StatefulDataLoader` requires the `torch` extra;
storage, formats, datasets, and I/O backends are available in the base package.

## How the pieces fit

A **storage format** encodes your records and decodes read results. A **storage**
class manages shards and reads documents or stream ranges. An **I/O backend**
reads byte ranges from individual shard files. A **dataset** maps sample indices
to storage reads, and the **DataLoader** turns those samples into training batches.

Readers and writers returned by storage and dataset factories are context managers:
use `with` for synchronous access and `async with` for asynchronous access.
The reference documents their public interfaces; private implementation classes
are internal. See the [compatibility policy](../compatibility.md) for release guarantees.
