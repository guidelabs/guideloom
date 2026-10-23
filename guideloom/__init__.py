__all__ = [
    "__version__",
    "DocumentDataset",
    "DocumentPickleFormat",
    "DocumentStorage",
    "StatefulDataLoader",
    "StreamDataset",
    "StreamNumpyFormat",
    "StreamStorage",
    "StreamWithMetaStorage",
    "io",
    "types",
]

from typing import TYPE_CHECKING

from . import io, types
from .dataset import DocumentDataset, StreamDataset
from .storage import DocumentStorage, StreamStorage, StreamWithMetaStorage
from .storage_format import DocumentPickleFormat, StreamNumpyFormat
from .version import __version__

if TYPE_CHECKING:
    from .pytorch import StatefulDataLoader as StatefulDataLoader


def __getattr__(name: str) -> object:
    if name == "StatefulDataLoader":
        from . import pytorch

        globals()[name] = pytorch.StatefulDataLoader
        return pytorch.StatefulDataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
