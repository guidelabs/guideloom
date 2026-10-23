import shutil
import pathlib

from tqdm import tqdm
import typed_settings as ts

from guideloom._benchmark import ConceptStorageFormat, FakeConceptData
from guideloom.storage import StreamWithMetaStorage


@ts.settings(frozen=True)
class DataEncoderSettings:
    persistent_dir: pathlib.Path = ts.option(
        default=pathlib.Path("."),
        help="Directory where all persistent data will be stored",
    )

    stride_start: int = ts.option(
        default=0,
        help="Which stride to start with",
    )

    stride_count: int = ts.option(
        default=3,
        help="How many strides to write",
    )

    clean: bool = ts.option(
        default=True,
        help="remove the dataset",
    )

    tokens_per_stride: int = ts.option(
        default=100_000_000,
        help="how many tokens per stride",
    )


@ts.cli(DataEncoderSettings, "data_encoder")
def main(args: DataEncoderSettings) -> None:
    storage_dir = args.persistent_dir / "preprocess_data"
    if args.clean:
        print("removing previous dataset")
        shutil.rmtree(storage_dir, ignore_errors=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    storage = StreamWithMetaStorage(storage_dir, ConceptStorageFormat())

    with storage.sync_writer() as writer:
        strides = range(args.stride_start, args.stride_start + args.stride_count)
        for stride in tqdm(strides, desc="stride", position=0):
            dataset = FakeConceptData(17, stride=stride, observations=args.tokens_per_stride)
            with tqdm(total=args.tokens_per_stride, position=1, leave=False, desc="token") as pbar:
                for obs in dataset:
                    writer.append(stride, obs)
                    pbar.update(writer.length(stride) - pbar.n)
                    if writer.length(stride) >= args.tokens_per_stride:
                        break


if __name__ == "__main__":
    main()
