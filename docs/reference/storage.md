# Custom storage

Create filesystem storage with the storage constructor, then open reader or writer
contexts. The abstract reader interfaces below document the methods returned by
these factories; private implementation classes are not part of this reference.

| Storage | Records | Format |
| --- | --- | --- |
| `StreamStorage` | Fixed-size elements read as windows across shards. | `StreamNumpyFormat` or a custom `StreamStorageFormat`. |
| `StreamWithMetaStorage` | Stream elements with variable-size annotations on spans. | Custom `StreamWithMetaStorageFormat`. |
| `DocumentStorage` | Separate variable-length documents. | `DocumentPickleFormat` or a custom `DocumentStorageFormat`. |

See [storage formats](formats.md) for encoding contracts and
[I/O backends](io.md) for the `read_interface` option.

## Storage constructors

::: guideloom.storage.StreamStorage
    options:
      members: [__init__, sync_writer, sync_reader, async_reader]

::: guideloom.storage.StreamWithMetaStorage
    options:
      members: [__init__, sync_writer, sync_reader, async_reader]

::: guideloom.storage.DocumentStorage
    options:
      members: [__init__, sync_writer, sync_reader, async_reader]

## Reader and writer interfaces

Use the returned objects in `with` blocks, or call `open()` and `close()`
explicitly. Readers expose `num_shards`, `total_length`, and `length(shard)`;
stream lengths count elements and document lengths count records. Writer
`length(shard)` counts what was appended since that writer was opened.
For awaitable reads, see [asynchronous storage readers](interfaces.md#asynchronous-storage-readers).

::: guideloom.types.StreamStorageSyncReaderAbstract
    options:
      members: [open, close, num_shards, total_length, length, read]
      inherited_members: true

::: guideloom.types.StreamStorageSyncWriterAbstract
    options:
      members: [open, close, append, length]
      inherited_members: true

::: guideloom.types.DocumentStorageSyncReaderAbstract
    options:
      members: [open, close, num_shards, total_length, length, read, read_many]
      inherited_members: true

::: guideloom.types.DocumentStorageSyncWriterAbstract
    options:
      members: [open, close, append, length]
      inherited_members: true
