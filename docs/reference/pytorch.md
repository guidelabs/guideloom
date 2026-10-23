# PyTorch DataLoader

Create a loader with `StatefulDataLoader.from_storage`. The loader owns its dataset
reader while open. Consume an iterator completely or close it before starting
another on the same loader. See [save and resume](../intermediate/save-resume.md)
for checkpoint lifecycle details.

Install with `uv pip install "guideloom[torch]"` and import the loader from
`guideloom` or `guideloom.pytorch`. Stream storage requires `sequence_length`;
document storage does not. `local_batch_size` is per rank, and
`global_batch_size` is `local_batch_size * world_size`. `len(loader)` reports
the number of batches per epoch, including the final partial global batch
when `drop_last=False`.

::: guideloom.pytorch.StatefulDataLoader
    options:
      members: [from_storage, open, close, __iter__, __len__, global_batch_size, state_dict, load_state_dict, get_default_device]

::: guideloom.types.StatefulDataLoaderStateDict
