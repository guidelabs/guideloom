from enum import StrEnum
import pathlib

from tqdm import tqdm
import torch
import typed_settings as ts

from guideloom._benchmark import ConceptStorageFormat
from guideloom.dataset import StreamDataset
from guideloom.io import MmapIO, PosixPreadIO, SeekIO
from guideloom.pytorch import StatefulDataLoader
from guideloom.storage import StreamWithMetaStorage


class ReadInterface(StrEnum):
    seek = "seek"
    mmap = "mmap"
    pread = "pread"


@ts.settings(frozen=True)
class DataEncoderSettings:
    persistent_dir: pathlib.Path = ts.option(
        default=pathlib.Path("."),
        help="Directory where all persistent data will be stored",
    )

    clean: bool = ts.option(
        default=True,
        help="remove the dataset",
    )

    sequence_length: int = ts.option(
        default=4096,
        help="How many tokens each observation should have",
    )

    epochs: int = ts.option(
        default=1,
        help="How many epochs to read",
    )

    batch_size: int = ts.option(
        default=2,
        help="Batch size",
    )

    rank: int = ts.option(
        default=0,
        help="Process rank",
    )

    world_size: int = ts.option(
        default=1,
        help="World size",
    )

    shuffle: bool = ts.option(
        default=True,
        help="Should shuffle dataset",
    )

    num_workers: int = ts.option(
        default=6,
        help="Number of dataloader workers",
    )

    prefetch_factor: int = ts.option(
        default=2,
        help="Prefetch factor",
    )

    async_read: bool = ts.option(
        default=True,
        help="Use sync or async read mode",
    )

    read_interface: ReadInterface = ts.option(
        default=ReadInterface.pread,
        help="How to interface with the file system",
    )


@ts.cli(DataEncoderSettings, "data_encoder")
def main(args: DataEncoderSettings) -> None:
    metal_interface = None
    match args.read_interface:
        case ReadInterface.pread:
            metal_interface = PosixPreadIO()
        case ReadInterface.mmap:
            metal_interface = MmapIO()
        case ReadInterface.seek:
            metal_interface = SeekIO()
        case _:
            raise ValueError("unreachable")

    storage_dir = args.persistent_dir / "preprocess_data"
    storage = StreamWithMetaStorage(storage_dir, ConceptStorageFormat(), read_interface=metal_interface)
    dataset = StreamDataset(storage, sequence_length=args.sequence_length)

    with StatefulDataLoader(
        dataset,
        collate_fn=lambda batch: batch,
        local_batch_size=args.batch_size,
        device=torch.device("cpu"),
        rank=args.rank,
        world_size=args.world_size,
        shuffle=args.shuffle,
        shuffle_seed=0,
        num_workers=args.num_workers,
        prefetch=args.num_workers * args.prefetch_factor,
    ) as dataloader:
        print(f"Observations per batch: {args.batch_size}")
        print(f"Total observations: {len(dataloader)}")
        print(f"Tokens per observations: {args.sequence_length}")
        print(f"Shuffle dataset: {args.shuffle}")

        for _epoch in tqdm(range(args.epochs), position=0, desc="epoch"):
            for _obs in tqdm(dataloader, position=1, desc="batch", leave=False):
                pass


if __name__ == "__main__":
    main()
