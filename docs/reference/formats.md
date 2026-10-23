# Storage formats

Formats define what `writer.append` accepts and what a reader returns. Pass a
format instance as `storage_format` to the matching storage constructor. The
`pack` method encodes one appended record; `unpack` decodes a requested document
or stream window.

## Built-in formats

Import these classes from `guideloom.storage_format` or directly from `guideloom`.

| Format | Use with | Encoding |
| --- | --- | --- |
| [`StreamNumpyFormat`](#guideloom.storage_format.StreamNumpyFormat) | `StreamStorage` | A mapping of equal-length, one-dimensional arrays, stored as fields of a structured NumPy dtype. |
| [`DocumentPickleFormat`](#guideloom.storage_format.DocumentPickleFormat) | `DocumentStorage` | One pickled Python object per document. |

`StreamNumpyFormat` requires a structured dtype whose field names match the input
mapping. `pack` builds a structured array; `unpack` returns field views of the
read array. Multiple fields can hold tokens, loss masks, or other aligned values.
The dtype is part of your schema: reopen storage with the same format.

`DocumentPickleFormat` takes no constructor arguments and uses `pickle.dumps`
and `pickle.loads`. Use it for trusted data; unpickling can execute code. For
numeric workloads, a custom binary format can avoid Python object reconstruction.
See [encoding tradeoffs](../advanced/performance.md#choose-an-encoding-that-matches-the-data).

::: guideloom.storage_format.StreamNumpyFormat
    options:
      members: [__init__, pack, unpack]

::: guideloom.storage_format.DocumentPickleFormat
    options:
      members: [pack, unpack]

## Custom format interfaces

Implement the interface matching your storage layout. Input and output types
can differ: a record appended during preprocessing need not have the same shape
as a window read during training.

| Interface | Encoding contract | Example |
| --- | --- | --- |
| `StreamStorageFormat` | `stream_dtype`, `pack`, and `unpack` define a fixed-size stream element and its decoded window. | [Custom stream format](../intermediate/streams.md#custom-stream-format) |
| `DocumentStorageFormat` | `pack` serializes a document; `unpack` decodes its bytes. | [Custom document format](../intermediate/documents.md#custom-document-format) |
| `StreamWithMetaStorageFormat` | `size` counts stream elements; `pack` yields metadata pairs and returns the stream array; `unpack` aligns metadata with the read window. | [Streams with metadata](../intermediate/stream-metadata.md) |

Streams with metadata require a custom format; there is no built-in metadata
codec. `StorageFormat[InT, OutT]` in `guideloom.types` is the union of these three
format interfaces.

::: guideloom.types.StreamStorageFormat
    options:
      members: [__init__, pack, unpack]

::: guideloom.types.StreamWithMetaStorageFormat
    options:
      members: [__init__, size, pack, unpack]

::: guideloom.types.DocumentStorageFormat
    options:
      members: [pack, unpack]
