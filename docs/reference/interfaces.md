# Extension interfaces

Import these interfaces from `guideloom.types`. They support custom storage,
dataset readers, and I/O backends. For codecs, see the
[format interfaces](formats.md#custom-format-interfaces). See also
[custom I/O backends](../advanced/io-backends.md) and
[asynchronous reads](../advanced/async-api.md).

## Storage factories

Implement one of these factories to provide storage to a dataset view or
`StatefulDataLoader.from_storage`. Stream offsets count stream elements;
document offsets count whole documents. Reader and writer methods are documented
under [Storage](storage.md#reader-and-writer-interfaces).

::: guideloom.types.StreamStorageAbstract
    options:
      members: [sync_writer, sync_reader, async_reader]

::: guideloom.types.DocumentStorageAbstract
    options:
      members: [sync_writer, sync_reader, async_reader]

## Dataset interfaces

Dataset factories return context-managed readers. Synchronous readers provide
`len(reader)` and `reader[index]`; asynchronous readers provide `len(reader)`
and `await reader[index]`. Both accept integer indices and slices. See
[Datasets](datasets.md) for the built-in views.

::: guideloom.types.DatasetAbstract
    options:
      members: [sync_reader, async_reader]

::: guideloom.types.DatasetSyncReaderAbstract
    options:
      members: [__len__, __getitem__]

::: guideloom.types.DatasetAsyncReaderAbstract
    options:
      members: [__len__, __getitem__]

## I/O backend interfaces

An `IOReaderAbstract` returns constructors for readers of individual files.
Those readers take a file path, support explicit open/close operations, and
return the requested byte range. The asynchronous reader also provides
`close_sync` for cleanup. See [built-in I/O backends](io.md) for implementations.

::: guideloom.types.IOReaderAbstract
    options:
      members: [is_supported, sync_reader, async_reader]

::: guideloom.types.IOSyncReaderAbstract
    options:
      members: [open, close, filesize, read]

::: guideloom.types.IOAsyncReaderAbstract
    options:
      members: [open, close, close_sync, filesize, read]

## Asynchronous storage readers

These readers have the same global offsets and shard metadata as their
[synchronous counterparts](storage.md#reader-and-writer-interfaces). Open them
with `async with` and await `read` or `read_many`; `num_shards`, `total_length`,
and `length(shard)` remain synchronous.

::: guideloom.types.StreamStorageAsyncReaderAbstract
    options:
      members: [read]

::: guideloom.types.DocumentStorageAsyncReaderAbstract
    options:
      members: [read, read_many]
