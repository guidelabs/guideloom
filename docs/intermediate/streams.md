# Streams

`StreamStorage` concatenates records into one logical stream. A read window can
cross both document and shard boundaries. The storage format defines each element.
Pass `sequence_length` to `StatefulDataLoader.from_storage` to choose how many
elements each training sample contains.

The example below opens a `StreamDataset` reader to show those windows directly.
That reader is part of the [dataset API](../advanced/dataset-api.md). A training
loop does not construct it.

## Tokens with a document ID and loss mask

This example stores three fixed-size fields per token. The repeated document ID
preserves provenance after concatenation; the mask can select which tokens
contribute to the training loss. All input arrays have the same length.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from guideloom import StreamDataset, StreamNumpyFormat, StreamStorage


dtype = np.dtype([("tokens", "<i4"), ("document_ids", "<u4"), ("loss_mask", "u1")])
with TemporaryDirectory() as directory:
    storage = StreamStorage(Path(directory), StreamNumpyFormat(dtype))
    with storage.sync_writer() as writer:
        writer.append(
            0,
            {
                "tokens": np.array([10, 11, 12], dtype="<i4"),
                "document_ids": np.full(3, 7, dtype="<u4"),
                "loss_mask": np.array([0, 1, 1], dtype="u1"),
            },
        )
        writer.append(
            1,
            {
                "tokens": np.array([13, 14, 15, 16, 17], dtype="<i4"),
                "document_ids": np.full(5, 8, dtype="<u4"),
                "loss_mask": np.array([0, 1, 1, 1, 1], dtype="u1"),
            },
        )

    with StreamDataset(storage, sequence_length=4).sync_reader() as reader:
        assert len(reader) == 2
        sample = reader[0]
        assert sample["tokens"].tolist() == [10, 11, 12, 13]
        assert sample["document_ids"].tolist() == [7, 7, 7, 8]
        assert sample["loss_mask"].tolist() == [0, 1, 1, 0]

    # Change the window length without rewriting the files.
    with StreamDataset(storage, sequence_length=3).sync_reader() as reader:
        assert len(reader) == 2  # The final two tokens do not fill a sample.
        assert reader[1]["tokens"].tolist() == [13, 14, 15]
```

These are [NumPy structured fields](numpy-structured-arrays.md) on
disk. Small fixed-size annotations fit directly beside tokens; they do not need
a separate metadata lookup. Document IDs must be globally unique if you use them
alone for provenance. A custom format can also return the shard segments passed
to `unpack(stream, shard_lengths, shards)`.

The loader shuffles **windows**, preserving token order within each window.
If you need to prevent cross-document windows, use [documents](documents.md)
or define a custom dataset view. For variable-length concept lists attached to
token spans, use [streams with metadata](stream-metadata.md).

## Custom stream format

Implement `StreamStorageFormat` when you need more control over packing or want
a different decoded output shape. `pack(data)` returns a one-dimensional NumPy
array with the declared `stream_dtype`. It may return a structured array directly,
such as tokens with a repeated document ID field.

`unpack(stream, shard_lengths, shards)` returns the sample. The `shard_lengths`
and `shards` arrays describe the consecutive shard segments making up the read.
Windows can cross both document and shard boundaries. Use these segments when
your output needs per-token shard IDs or other provenance.

Use explicit byte order and the same dtype when writing and reading. Preserve
the encoding and decoding code with the dataset. See the
[format API reference](../reference/formats.md) for method signatures.
