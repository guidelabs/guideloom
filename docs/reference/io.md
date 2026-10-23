# Built-in I/O backends

Import these backends from `guideloom.io` and pass an instance as `read_interface`
to a storage constructor. They control how shard bytes are read; they do not
change the stored format or the write path. All three provide synchronous and
asynchronous reader factories.

| Backend | Read behavior | Availability |
| --- | --- | --- |
| [`SeekIO`](#guideloom.io.SeekIO) | Opens, seeks, and reads into a writable buffer for each request. Default when `read_interface` is omitted. | All supported platforms. |
| [`MmapIO`](#guideloom.io.MmapIO) | Keeps a read-only memory map and copies each requested range into a writable buffer. | All supported platforms. |
| [`PosixPreadIO`](#guideloom.io.PosixPreadIO) | Keeps a file descriptor and reads directly into a writable buffer at an explicit offset using `os.preadv`. | POSIX with `os.preadv`; check `is_supported()`. |

For `PosixPreadIO(advice=True)`, check `is_supported(advice=True)` first. This also
requires `posix_fadvise` and requests `POSIX_FADV_RANDOM` when opening files.

## Choose a backend

This example falls back to the portable default when positional reads are
unavailable:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from guideloom import StreamNumpyFormat, StreamStorage
from guideloom.io import PosixPreadIO, SeekIO

backend = PosixPreadIO() if PosixPreadIO.is_supported() else SeekIO()
with TemporaryDirectory() as directory:
    storage = StreamStorage(
        Path(directory),
        StreamNumpyFormat(np.dtype([("tokens", "<i4")])),
        read_interface=backend,
    )
    with storage.sync_writer() as writer:
        writer.append(0, {"tokens": np.arange(8, dtype=np.int32)})
    with storage.sync_reader() as reader:
        assert reader.read(2, 4)["tokens"].tolist() == [2, 3, 4, 5]
```

See [performance tips](../advanced/performance.md#select-a-read-backend-pread-seek-or-mmap)
for tuning. To implement another backend, see
[custom I/O backends](../advanced/io-backends.md) and the
[I/O reader interfaces](interfaces.md#io-backend-interfaces).

## Backend API

`sync_reader()` and `async_reader()` return reader constructors, which accept a
file path. Storage opens and closes those readers for you. The returned readers
expose `filesize()` and `read(offset, size)` through the
[reader interfaces](interfaces.md#io-backend-interfaces).

::: guideloom.io.SeekIO
    options:
      members: [is_supported, sync_reader, async_reader]

::: guideloom.io.MmapIO
    options:
      members: [is_supported, sync_reader, async_reader]

::: guideloom.io.PosixPreadIO
    options:
      members: [__init__, is_supported, sync_reader, async_reader]
