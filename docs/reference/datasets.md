# Datasets

Dataset views provide indexed samples. PyTorch training uses
`StatefulDataLoader.from_storage`, which creates these views for you. See
[the dataset API](../advanced/dataset-api.md) to open them directly.

| View | Sample at index `i` |
| --- | --- |
| `StreamDataset(storage, sequence_length)` | A complete, non-overlapping window starting at `i * sequence_length`. Trailing elements that do not fill a window are omitted. |
| `DocumentDataset(storage)` | The complete document at global document index `i`. |

Open `sync_reader()` with `with`, then use `len(reader)`, `reader[i]`, or
`reader[start:stop]`. For async access, use `async with dataset.async_reader()`
and await the indexed result. Negative integer indices raise `IndexError`;
slices use normal slice semantics. The
[dataset reader interfaces](interfaces.md#dataset-interfaces) document these operations.

::: guideloom.dataset.StreamDataset
    options:
      members: [__init__, sync_reader, async_reader]

::: guideloom.dataset.DocumentDataset
    options:
      members: [__init__, sync_reader, async_reader]
