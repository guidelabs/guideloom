# Documents

Use `DocumentStorage` when each training sample should remain one whole record,
such as a conversation. A document can contain tokens, labels, IDs, or any other
fields your format can encode. Each sample is one decoded record. Records are
not concatenated or padded. Pass the storage to `StatefulDataLoader.from_storage`
and omit `sequence_length`.

## Store trusted Python records

This self-contained example writes two documents and pads them in the PyTorch
collator. Real preprocessing and training should be separate scripts, as in the
[quick-start](../quick-start.md).

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn.utils.rnn import pad_sequence

from guideloom import DocumentPickleFormat, DocumentStorage, StatefulDataLoader


def collate(samples):
    lengths = torch.tensor([len(sample["tokens"]) for sample in samples])
    tokens = pad_sequence(
        [torch.tensor(sample["tokens"], dtype=torch.int64) for sample in samples],
        batch_first=True,
        padding_value=0,
    )
    return {"tokens": tokens, "lengths": lengths}


with TemporaryDirectory() as directory:
    storage = DocumentStorage(Path(directory), DocumentPickleFormat())
    with storage.sync_writer() as writer:
        writer.append(0, {"document_id": "a", "tokens": [10, 11]})
        writer.append(1, {"document_id": "b", "tokens": [12, 13, 14]})

    with storage.sync_reader() as reader:
        assert reader.read(1)["document_id"] == "b"

    with StatefulDataLoader.from_storage(
        storage, collate, device=torch.device("cpu"), local_batch_size=2
    ) as loader:
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["tokens"].tolist() == [[10, 11, 0], [12, 13, 14]]
        assert batches[0]["lengths"].tolist() == [2, 3]
```

Pickle can execute code when decoded; only load pickle datasets you trust. It is
convenient for prototypes, but constructing many Python objects can be expensive.
The [custom document format](#custom-document-format)
uses `struct` and `np.frombuffer` to return a document ID, token array, and shard
number without pickle. JSON or MessagePack are other options for general records;
see [performance tips](../advanced/performance.md).


## Custom document format

A format defines both the input you write and the decoded object you read. These
shapes can differ: the following format stores a document ID and tokens, then
also returns the source shard when decoding. It's best to keep the format class
in an importable module shared by preprocessing and training.

Implement `DocumentStorageFormat.pack(data)` to return bytes and
`unpack(encoded_data, shard)` to decode them. This example stores a four-byte
document ID followed by little-endian 32-bit tokens, without pickle.

```python
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import numpy as np

from guideloom import DocumentStorage
from guideloom.types import DocumentStorageFormat


# Typing is optional. These TypedDicts describe the shapes for a type checker.
class TokenDocument(TypedDict):
    document_id: int
    tokens: list[int]


class DecodedTokenDocument(TypedDict):
    document_id: int
    tokens: np.ndarray
    shard: int


# The type parameters and method annotations below are optional.
class TokenDocumentFormat(DocumentStorageFormat[TokenDocument, DecodedTokenDocument]):
    prefix = struct.Struct("<I")

    def pack(self, data: TokenDocument) -> bytes:
        return self.prefix.pack(data["document_id"]) + np.asarray(data["tokens"], dtype="<i4").tobytes()

    def unpack(self, encoded_data, shard: int) -> DecodedTokenDocument:
        (document_id,) = self.prefix.unpack_from(encoded_data)
        tokens = np.frombuffer(encoded_data, dtype="<i4", offset=self.prefix.size)
        return {"document_id": document_id, "tokens": tokens, "shard": shard}


with TemporaryDirectory() as directory:
    storage = DocumentStorage(Path(directory), TokenDocumentFormat())
    with storage.sync_writer() as writer:
        writer.append(0, {"document_id": 7, "tokens": [10, 11, 12]})
    with storage.sync_reader() as reader:
        assert reader.read(0)["tokens"].tolist() == [10, 11, 12]
```

Keep the codec and schema with the dataset. Your application owns compatibility
of custom encodings; package version guarantees cannot validate changes to your
dtype, byte order, payload schema, or decoding code.
