# Save and resume

Capture `loader.state_dict()` after processing a batch, alongside your model,
optimizer, and training RNG state. It records the global sample position delivered
to the consumer, so prefetched batches do not advance the saved position.

## Save and restore the state

This complete example saves the loader state as JSON, resumes in a fresh loader,
and compares the remaining batches. In a training job, put this state in the same
checkpoint as your model and optimizer after the training step completes.

Workers and prefetching are enabled to show that prepared batches do not move
the saved position ahead of the consumer.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import torch

from guideloom import DocumentPickleFormat, DocumentStorage, StatefulDataLoader


def collate(samples):
    return list(samples)


with TemporaryDirectory() as directory:
    storage = DocumentStorage(Path(directory), DocumentPickleFormat())
    with storage.sync_writer() as writer:
        for value in range(12):
            writer.append(0, value)
    settings = dict(
        collate_fn=collate,
        device=torch.device("cpu"),
        local_batch_size=2,
        shuffle=True,
        shuffle_seed=42,
        num_workers=2,
        prefetch=2,
    )
    with StatefulDataLoader.from_storage(storage, **settings) as loader:
        iterator = iter(loader)
        first_batch = next(iterator)
        checkpoint = loader.state_dict()
        assert checkpoint["observation_offset"] == 2
        checkpoint_path = Path(directory) / "loader-state.json"
        checkpoint_path.write_text(json.dumps(checkpoint))
        expected_remaining = list(iterator)

    with StatefulDataLoader.from_storage(storage, **settings) as resumed:
        resumed.load_state_dict(json.loads(checkpoint_path.read_text()))
        assert list(resumed) == expected_remaining
```

Use a fresh loader for resumption. Closing an iterator resets that loader's sample
position, so capture the state before closing it. `load_state_dict` also rejects
restoration while the target loader has an active iterator.

## What the checkpoint contains

A loader checkpoint contains `epoch`, `observation_offset`, `shuffle`, `shuffle_seed`,
and `drop_last`. The offset advances by `global_batch_size` for each delivered batch,
including padding in an incomplete final batch. The epoch and offset identify the
position already delivered to the consumer across all ranks. It does not store
prefetched samples, dataset files, model state, optimizer state, or random state
used by transformations.

The loader warns when the saved shuffle/drop settings differ from its
configuration. It does not record or validate a package version, dataset identity,
dataset size, sequence length, batch size, or world size. Save these settings in
your training application when you need to verify them on resumption.

## Preserve sample order

The loader reconstructs its position from the epoch, global observation offset,
shuffle seed, and dataset size. You can change the local batch size or world size
when resuming: the remaining global sample order is preserved, with samples
repartitioned across the new ranks and batches. Load the same checkpoint on every
rank after all original ranks have completed the same number of steps. Individual
ranks can receive different samples than they would have before the change.

For the same remaining sample order, preserve the dataset contents and ordering,
storage format, dataset view and sequence length, `shuffle`, `shuffle_seed`, and
`drop_last`. With `drop_last=True`, the new global batch size determines
how many samples at the end of the remaining sequence are dropped. With
`drop_last=False`, all remaining samples are delivered, and some ranks can receive
an empty final batch.

Changes to worker count, prefetching, or device placement should not change sample
ordering. Identical tensor values still depend on your transforms, RNG state,
hardware, and dependency behavior. A loader checkpoint is not a snapshot of user
code or dataset contents.

See [compatibility](../compatibility.md) for guarantees across package releases.
