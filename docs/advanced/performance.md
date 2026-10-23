# Performance tips

Measure on the filesystem, sequence length, and payload format you plan to use.
Separate first-batch latency from steady-state throughput: compiled index
calculations may need a first-use compilation, and filesystem caches change
subsequent reads. Include collation and device transfers in end-to-end timings.

## Select a read backend: pread, seek, or mmap

`SeekIO` is the portable default. It opens a file, seeks, and reads for each
request. On platforms with `os.preadv`, `PosixPreadIO` keeps a descriptor open
and reads into a buffer at an explicit offset, avoiding a shared seek position.
Check support through `is_supported()`; POSIX alone does not imply `preadv` exists.

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

On supported systems, `PosixPreadIO(advice=True)` also requests random-access
advice using `posix_fadvise`; check `is_supported(advice=True)` first. Benchmark
this against the default on shared filesystems, where kernel readahead and
caching policies can matter. `MmapIO` is another option for mapped reads;
its page-fault and cache behavior depends on the workload.

Backends keep resources for accessed shards, so account for file-descriptor
limits when choosing shard counts. Shards should be large enough to avoid many
tiny files while still allowing useful preprocessing concurrency.

## Tune workers and prefetch together

Start with `num_workers=0` and `prefetch=0` to establish a baseline. Then try a
small thread pool and increase it while measuring training throughput. With
multiple ranks on a node, the total loading concurrency scales with both ranks
and workers; reserve CPU and storage bandwidth for training too.

`prefetch` counts batches, not samples. Keep
`prefetch * local_batch_size >= num_workers` to give every loading worker useful
work; the loader warns otherwise. More prefetching also holds more decoded data
and batches in memory. Increase it only while it hides latency usefully.

Threads share buffers and avoid process serialization. `num_workers` threads
shuffle and read. One thread batches, collates, pins memory, and fills the
prefetch queue. The training thread moves that batch onto the device. Keep
functions at module scope.
Vectorized NumPy operations or suitable compiled transforms can reduce Python
work instead. Custom pack/unpack code is not automatically GIL-free just because
GuideLoom compiles its own indexing.

## Choose an encoding that matches the data

| Data | Encoding to consider | Tradeoff |
| --- | --- | --- |
| Fixed-size numeric fields | Structured NumPy arrays | Direct buffer views; stable dtype required. |
| Numeric vectors in documents or metadata | `array.tobytes()` and `np.frombuffer()` | Record dtype and shape; add a small header if needed. |
| Small fixed headers | `struct.Struct` | Explicit binary layout without constructing a large object tree. |
| Variable nested records | MessagePack or JSON in a custom format | Flexible, but parsing and object construction still cost time. |
| Arbitrary trusted Python objects | Pickle | Convenient, with Python object reconstruction and GIL costs. |

`DocumentPickleFormat` is useful for getting started, but avoid making it the
default for a high-throughput numeric dataset. MessagePack requires an additional
codec dependency and a defined schema; neither it nor JSON automatically makes
numeric payloads zero-copy. The [custom document format](../intermediate/documents.md#custom-document-format)
combines `struct` with a NumPy buffer view. For metadata, encode once per contiguous
span and yield `None` for subsequent tokens in that span.

## Account for copies and device placement

`np.frombuffer` can expose the storage read buffer without an extra decode copy.
Packing, joining data across shards, stacking samples, pinning memory, and moving
to an accelerator can still allocate or copy. A structured field may have a
non-contiguous stride; make a contiguous batch once in the collator rather than
converting each token separately.

For CUDA, select the current process's GPU explicitly and use `pin_memory=True`.
The loader moves tensors to its configured device after CPU prefetching. Copies
from CPU onto CUDA are queued with `non_blocking=True`, so the host does not wait
on each tensor; later work on that CUDA stream still runs after the copy. The
loader does not maintain a separate CUDA-stream prefetch queue. Measure transfer
overlap in your training setup before assuming it is hidden. CPU examples do not
validate CUDA throughput. Always wait for device work when timing it.
