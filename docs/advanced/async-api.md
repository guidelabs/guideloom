# Asynchronous reads

Storage and [dataset views](dataset-api.md) expose `async_reader()` as well as
`sync_reader()`. PyTorch training uses the synchronous `StatefulDataLoader.from_storage`
instead. Use `async with` to open resources. Dataset indexing returns an
awaitable; `len(reader)` remains synchronous after opening.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from guideloom import DocumentDataset, DocumentPickleFormat, DocumentStorage


async def main():
    with TemporaryDirectory() as directory:
        storage = DocumentStorage(Path(directory), DocumentPickleFormat())
        with storage.sync_writer() as writer:
            for value in range(4):
                writer.append(0, value)
        async with storage.async_reader() as reader:
            assert await reader.read(1) == 1
        async with DocumentDataset(storage).async_reader() as reader:
            assert len(reader) == 4
            assert await reader[1] == 1
            assert list(await reader[1:3]) == [1, 2]
            # Independent requests can be awaited together.
            assert await asyncio.gather(reader[0], reader[3]) == [0, 3]


if __name__ == "__main__":
    asyncio.run(main())
```

For streams, await `reader.read(global_index, global_length)`. The serialization
format methods remain synchronous. There is currently no public asynchronous
writer API or asynchronous DataLoader; use a sync writer for preprocessing and
the sync loader for training.

Bound concurrency when scheduling large numbers of reads; gathering an entire
corpus at once defeats streaming. Stream reads can gather the shard ranges
needed for one window, while a custom remote backend can overlap network
requests. See [custom I/O backends](io-backends.md#http-range-requests).
