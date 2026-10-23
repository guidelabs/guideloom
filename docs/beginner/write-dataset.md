# Write a dataset

Preprocessing writes a persistent dataset. Training later opens the completed
files for reading. First [choose a storage format](storage.md);
this guide uses the token stream from the [quick-start](../quick-start.md).

A storage object describes the directory and encoding. Opening `sync_writer()`
creates a writer, and `append(shard, data)` adds a record to that shard during
that writer session. Closing the context finishes the writes.

## Write independent shards in parallel

Give each shard exactly one owner. The following preprocessing script writes
four shards using two processes. Each task creates its own storage object and
writer, and owns a distinct shard number.

Save as `write_shards.py` and run `python write_shards.py` from your environment.

```python title="write_shards.py"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from guideloom import StreamNumpyFormat, StreamStorage


DIRECTORY = Path("data/sharded-tokens")


def write_shard(shard):
    storage = StreamStorage(DIRECTORY, StreamNumpyFormat(np.dtype([("tokens", "<i4")])))
    with storage.sync_writer() as writer:
        # Replace this loop with preprocessing for this shard's input records.
        for record in range(2):
            start = shard * 16 + record * 8
            writer.append(shard, {"tokens": np.arange(start, start + 8, dtype=np.int32)})
    return shard


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(write_shard, range(4)))
    assert completed == [0, 1, 2, 3]
    print(f"Finished {len(completed)} shards in {DIRECTORY}")


if __name__ == "__main__":
    main()
```

The task's `shard` argument determines its output files. A process may write
several shards sequentially, but no two tasks may write the same shard. With a
cluster scheduler, assign disjoint shard numbers to jobs in the same way.
GuideLoom leaves task scheduling and source-data partitioning to your application.
Use a script with an entry-point guard for multiprocessing.

## Keep shard ownership and publication explicit

- Number completed shards consecutively from zero: `0, 1, 2, ...`, without gaps.
- Append repeatedly through the same open writer to extend that shard. A new
  writer that writes an existing shard truncates its previous files.
- Wait for every writer to finish before starting readers. Keep the dataset
  unchanged while training or resuming a checkpoint.
- Use a new directory for a new preprocessing run. When retrying a failed shard,
  rewrite all of that shard through one owner before publishing the dataset.

The same ownership rule applies to `DocumentStorage` and
`StreamWithMetaStorage`, including their index and payload files. The number of
preprocessing workers and shards is independent of the training world size.

Next, [read the dataset into PyTorch](load-batches.md). For other layouts, see
[documents](../intermediate/documents.md) and [streams with metadata](../intermediate/stream-metadata.md).
