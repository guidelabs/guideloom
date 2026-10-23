# Streams with metadata

Use `StreamWithMetaStorage` when tokens have variable-size annotations, such as
concept lists attached to paragraphs. The token stream stays contiguous; a
separate payload stores each span's annotation. A window can start or end inside
a span, or cross document and shard boundaries, while retaining the mapping.
Training passes this storage to `StatefulDataLoader.from_storage` with a
`sequence_length`. The example opens a `StreamDataset` reader only to show one
window; see [the dataset API](../advanced/dataset-api.md).

## Tokens with chunk concepts and provenance

The custom format below stores 32-bit tokens as **stream** data. Its **document**
payload is a binary document ID and chunk ID followed by a variable-length array
of concept IDs. This is the kind of span annotation that a fixed-size stream
field cannot represent directly.

```python
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from guideloom import StreamDataset, StreamWithMetaStorage
from guideloom.types import StreamWithMetaStorageFormat


class ConceptFormat(StreamWithMetaStorageFormat[dict, dict, np.int32]):
    prefix = struct.Struct("<II")
    concept_dtype = np.dtype("<i4")

    def __init__(self):
        super().__init__(np.dtype("<i4"))

    def size(self, data):
        return len(data["tokens"])

    def pack(self, data):
        tokens = np.asarray(data["tokens"], dtype=self.stream_dtype)
        assert len(data["chunk_ids"]) == len(tokens)
        for chunk_id in data["chunk_ids"]:
            payload = self.prefix.pack(data["document_id"], chunk_id)
            payload += np.asarray(data["concepts"][chunk_id], dtype=self.concept_dtype).tobytes()
            yield ((data["document_id"], chunk_id), payload)
        return tokens

    def unpack(self, stream_data, document_shard, document_index, document_data):
        annotations = []
        for shard, payload in zip(document_shard, document_data, strict=True):
            document_id, chunk_id = self.prefix.unpack_from(payload)
            annotations.append(
                {
                    "shard": int(shard),
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "concepts": np.frombuffer(payload, dtype=self.concept_dtype, offset=self.prefix.size),
                }
            )
        return {"tokens": stream_data, "annotation_index": document_index, "annotations": annotations}


with TemporaryDirectory() as directory:
    storage = StreamWithMetaStorage(Path(directory), ConceptFormat())
    with storage.sync_writer() as writer:
        writer.append(
            0,
            {
                "document_id": 7,
                "tokens": [10, 11, 12],
                "chunk_ids": [0, 0, 1],
                "concepts": [[100, 101], [102]],
            },
        )
        writer.append(
            1,
            {
                "document_id": 8,
                "tokens": [13, 14, 15, 16, 17],
                "chunk_ids": [0, 0, 0, 0, 0],
                "concepts": [[200, 201, 202]],
            },
        )

    with StreamDataset(storage, sequence_length=4).sync_reader() as reader:
        sample = reader[0]
        assert sample["tokens"].tolist() == [10, 11, 12, 13]
        per_token = [sample["annotations"][i] for i in sample["annotation_index"]]
        assert [a["document_id"] for a in per_token] == [7, 7, 7, 8]
        assert per_token[-1]["shard"] == 1
        assert per_token[-1]["concepts"].tolist() == [200, 201, 202]
```

## The packing contract

`size(data)` gives the number of stream elements. `pack(data)` must yield exactly
one `(key, payload)` pair per element, then **return** the one-dimensional array
with the declared dtype and length. Returning the array from the generator is
different from yielding it.

Equal consecutive keys within a record share one stored payload. Keys must
identify contiguous runs: `[0, 0, 1, 1]` is valid, but `[0, 1, 0]` is not. The
first occurrence of each key needs a payload; later elements in that run may
yield `None` instead of encoding the same payload again. The example repeats
encoding for clarity; production formats can encode each run once.

Keys are local to an `append` call and are not stored as application identifiers.
Include document and chunk IDs in the payload when you need them for tracing.

## The unpacking contract

`document_index` has one entry per returned token. Each entry indexes
`document_data`, the payloads needed by this read. `document_shard` gives
the source shard for each payload. These indices are remapped for every read;
they are not the original chunk IDs. Decode the payload's own IDs to recover
provenance, as the example does.

A metadata format can also use a [structured dtype](numpy-structured-arrays.md)
for multiple stream fields. For a small fixed-size value such as one document
ID per token, [plain stream fields](streams.md) avoid the extra payload lookup.
