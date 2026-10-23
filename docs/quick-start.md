# Quick-start

Use Python 3.13 or 3.14 and install GuideLoom with PyTorch support:

```sh
uv venv --python 3.13
uv pip install "guideloom[torch]"
```

For an unreleased checkout, use `uv pip install -e ".[torch]"` instead.
Run the scripts below with that environment's Python.

A dataset has two separate stages: **write it during preprocessing**, then **open
it for training later**. Save these files in the same working directory. The
`data/quick-start` directory persists between runs.

## 1. Declare your format

Save the shared format in `dataset_format.py`. Both preprocessing and training
import this definition so they use the same field names and binary layout.

```python title="dataset_format.py"
import numpy as np

from guideloom import StreamNumpyFormat


storage_format = StreamNumpyFormat(np.dtype([("tokens", "<i4")]))
```

`tokens` names the array being stored. `<i4` means a little-endian, four-byte
integer. The [structured-array introduction](intermediate/numpy-structured-arrays.md)
explains how to add more fields.

## 2. Write the dataset

Save as `write_dataset.py`. This small example stands in for tokenization or
another preprocessing job.

```python title="write_dataset.py"
from pathlib import Path

import numpy as np

from dataset_format import storage_format
from guideloom import StreamStorage


def main():
    directory = Path("data/quick-start")
    directory.mkdir(parents=True, exist_ok=False)
    storage = StreamStorage(directory, storage_format)
    with storage.sync_writer() as writer:
        writer.append(0, {"tokens": np.arange(16, dtype=np.int32)})
    print(f"Dataset written to {directory}")


if __name__ == "__main__":
    main()
```

```sh
python write_dataset.py
```

Use a new output directory when rerunning preprocessing. Opening an existing
shard for writing replaces that shard's contents.

## 3. Read it later into PyTorch

Save as `train.py`. It reopens the existing files using the same format; it does
not run the preprocessing script.

```python title="train.py"
from pathlib import Path

import numpy as np
import torch

from dataset_format import storage_format
from guideloom import StatefulDataLoader, StreamStorage


def collate(samples):
    return torch.from_numpy(np.stack([sample["tokens"] for sample in samples]))


def main():
    storage = StreamStorage(Path("data/quick-start"), storage_format)
    with StatefulDataLoader.from_storage(
        storage,
        collate,
        sequence_length=4,
        device=torch.device("cpu"),
        local_batch_size=2,
        shuffle=True,
        shuffle_seed=0,
    ) as loader:
        for batch in loader:
            print(batch)
            assert batch.shape == (2, 4)


if __name__ == "__main__":
    main()
```

```sh
python train.py
```

The 16 stored tokens become four non-overlapping, four-token samples, delivered
in two batches. `sequence_length` is chosen when reading, so you can change the
context length without rewriting the stream. A trailing partial sample is omitted.
The collator stacks samples into a tensor; replace the print with your training step.

Next, [choose a storage format](beginner/storage.md),
[write shards in parallel](beginner/write-dataset.md), and
[load batches across distributed ranks](beginner/load-batches.md).
