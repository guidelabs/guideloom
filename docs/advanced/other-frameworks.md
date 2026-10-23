# Use another framework, such as JAX

Storage and dataset readers return the objects defined by your format, usually
NumPy arrays. Install the base package with `uv pip install guideloom` when
another framework places those arrays on device. PyTorch training should use
`StatefulDataLoader.from_storage` from the `torch` extra
(`uv pip install "guideloom[torch]"`). Other frameworks read through the
[dataset API](dataset-api.md).

## A single-process JAX loop

Install JAX for your target hardware following its
[installation guide](https://docs.jax.dev/en/latest/installation.html).
The following self-contained CPU example shows the handoff. In a training job,
write the data separately and reopen it using the same format, as in the
[quick-start](../quick-start.md).

```python title="train_jax.py"
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np

from guideloom import StreamDataset, StreamNumpyFormat, StreamStorage


with TemporaryDirectory() as directory:
    storage = StreamStorage(Path(directory), StreamNumpyFormat(np.dtype([("tokens", "<i4")])))
    with storage.sync_writer() as writer:
        writer.append(0, {"tokens": np.arange(16, dtype=np.int32)})

    dataset = StreamDataset(storage, sequence_length=4)
    local_batch_size = 2
    with dataset.sync_reader() as reader:
        stop = len(reader) // local_batch_size * local_batch_size
        for start in range(0, stop, local_batch_size):
            samples = reader[start : start + local_batch_size]
            host_batch = np.stack([sample["tokens"] for sample in samples])
            batch = jax.device_put(host_batch)
            assert batch.shape == (2, 4)
            # Pass batch to your JAX training step here.
```

`jax.device_put` accepts NumPy arrays and transfers them to JAX's default device.
Device transfer and execution can be asynchronous; synchronize when measuring
throughput. See the [JAX data-placement API](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).

## Own ordering and checkpoints in a custom loop

A direct reader does not shuffle, batch, prefetch, or save a training position.
Your loop owns those steps. The following framework-independent example tests
one epoch's ordering across two simulated ranks. It uses GuideLoom's public
`FeistelShuffler` to map global positions to samples without a stored permutation.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from guideloom import StreamDataset, StreamNumpyFormat, StreamStorage
from guideloom.shuffle import FeistelShuffler


with TemporaryDirectory() as directory:
    storage = StreamStorage(Path(directory), StreamNumpyFormat(np.dtype([("tokens", "<i4")])))
    with storage.sync_writer() as writer:
        writer.append(0, {"tokens": np.arange(32, dtype=np.int32)})

    with StreamDataset(storage, sequence_length=4).sync_reader() as reader:
        local_batch_size, world_size = 2, 2
        global_batch_size = local_batch_size * world_size
        epoch, batch_step, seed = 0, 0, 42
        rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(seed, spawn_key=(epoch,))))
        shuffler = FeistelShuffler(len(reader), rng=rng)
        stop = len(reader) // global_batch_size * global_batch_size
        sample_ids = []
        for rank in range(world_size):
            positions = range(batch_step * global_batch_size + rank, stop, world_size)
            ids = [shuffler.shuffle(position) for position in positions]
            sample_ids.extend(ids)
            for start in range(0, len(ids), local_batch_size):
                batch = np.stack([reader[i]["tokens"] for i in ids[start : start + local_batch_size]])
                assert batch.shape == (2, 4)
        assert sorted(sample_ids) == list(range(len(reader)))
```

For pure data parallel JAX, replace the simulated ranks with `jax.process_index()`
and `jax.process_count()` after distributed initialization. The index arithmetic
above is one possible partitioning policy for your loop, rather than a built-in
JAX loader. Ensure the local batch size fits the devices assigned to that process.
Create global arrays with the appropriate sharding, for example using
`jax.make_array_from_process_local_data`; follow JAX's
[distributed data-loading guide](https://docs.jax.dev/en/latest/distributed_data_loading.html)
for data and model parallel layouts.

Save the epoch, completed global batch step, shuffle seed, dataset view, and
partition settings alongside the training checkpoint. Recreate the same shuffler
and start positions when resuming. Advance the saved step only after consuming
the batch, even if you add your own prefetch queue. This loop's state is your
application's contract; it does not use `StatefulDataLoader.state_dict()`.
