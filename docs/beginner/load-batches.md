# Read into PyTorch

Open a completed dataset using the storage class and format used during
preprocessing, then pass that storage to `StatefulDataLoader.from_storage`.
Stream storage, including streams with metadata, requires `sequence_length`.
Document storage omits it. A `collate_fn` turns samples into a batch; the loader
handles ordering, workers, and devices.

Construct a `StreamDataset` or `DocumentDataset` yourself only when you need a
custom dataset or indexed access outside this loader. See
[the dataset API](../advanced/dataset-api.md).

The [quick-start](../quick-start.md) shows a single-process CPU reader. This
example reads the four shards from [write a dataset](write-dataset.md) using two
training processes and two loading threads per process.

## Distributed CPU example

Save as `train_distributed.py`. Write the shards first, then run:

```sh
python write_shards.py
torchrun --standalone --nproc-per-node=2 train_distributed.py
```

```python title="train_distributed.py"
import os
from pathlib import Path

import numpy as np
import torch

from guideloom import StatefulDataLoader, StreamNumpyFormat, StreamStorage


def collate(samples):
    return torch.from_numpy(np.stack([sample["tokens"] for sample in samples]))


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    storage = StreamStorage(Path("data/sharded-tokens"), StreamNumpyFormat(np.dtype([("tokens", "<i4")])))
    with StatefulDataLoader.from_storage(
        storage,
        collate,
        sequence_length=4,
        device=torch.device("cpu"),
        local_batch_size=2,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        shuffle_seed=42,
        num_workers=2,
        parallel_method="thread",
        prefetch=2,
    ) as loader:
        for batch in loader:
            assert batch.shape == (2, 4)
            print(f"rank={rank}: {batch.tolist()}")


if __name__ == "__main__":
    main()
```

The environment defaults also let you run this as `python train_distributed.py`
for a single rank. With two ranks, each gets eight of the sixteen samples: four
local batches each. This demonstrates loading; initialize your training
framework's distributed model and optimizer separately.

## Rank and world size

Pass the **global rank**, from zero to `world_size - 1`, and the total number of
training processes. GuideLoom does not discover these values. Every rank must
use the same dataset, sequence length, batch size, shuffle seed, and epoch.

The global batch size is `local_batch_size * world_size`. Each rank takes its
own strided positions in the global index sequence; a seeded shuffle maps those
positions to samples across all shards. Loaders do not communicate with each
other, and each rank reads its own samples directly from the dataset files.

`drop_last=True` is the default: the incomplete final global batch is omitted.
With `drop_last=False`, ranks take the same number of steps, but their final
local batches can have different sizes. A rank can receive no samples in that
step, so its collator must handle `[]`. Ensure the training loop also supports
those uneven batches. The dataset must contain at least one full global batch
when the loader opens, even with `drop_last=False`.

## Loading workers and GPUs

`num_workers` is per training process and is separate from `world_size`. Start
with zero workers when debugging. Those threads shuffle and read. One more thread
batches, collates, and prefetches so that work overlaps training; see
[performance tips](../advanced/performance.md). The training thread moves the
batch onto the device.

For one CUDA device per process, choose `torch.device("cuda", local_rank)` from
your training framework's **local** rank and set `pin_memory=True`. The global
rank still controls data partitioning. Set the device explicitly to avoid
accidentally sending multiple processes to the same GPU.

Each fully consumed iterator advances the epoch. Drain or close an iterator
before creating another on the same loader. To stop mid-epoch and continue
later, [save the state before closing it](../intermediate/save-resume.md).
