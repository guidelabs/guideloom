from typing import TypedDict


class StatefulDataLoaderStateDict(TypedDict):
    """Checkpoint returned by `StatefulDataLoader.state_dict`.

    `epoch` and `observation_offset` identify the global sample position delivered to the consumer.
    Prefetched batches that have not been yielded are not included. `shuffle`, `shuffle_seed`, and
    `drop_last` record how the loader was constructed. They are not applied by `load_state_dict`.

    Attributes:
        epoch (int): Epoch index. It increments when an iterator is exhausted.
        observation_offset (int): Global sample position in the current epoch. Advances by the global
            batch size for each delivered batch, including padding in an incomplete final batch.
        shuffle (bool): Whether that loader was constructed with shuffling enabled.
        shuffle_seed (int | None): Seed used for shuffling. `None` is not reproducible.
        drop_last (bool): Whether that loader drops an incomplete global batch.
    """

    epoch: int
    observation_offset: int
    shuffle: bool
    shuffle_seed: int | None
    drop_last: bool
