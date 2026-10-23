# Custom I/O backends

An I/O backend reads byte ranges from shard files. Pass one as
`read_interface` to a storage constructor. The built-in
choices are `SeekIO` (the default), `MmapIO`, and `PosixPreadIO`.
Check platform support before selecting an optional backend. Memory mapping can
be useful for repeated random access, but benchmark with your actual files and
storage device.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from guideloom import DocumentPickleFormat, DocumentStorage
from guideloom.io import MmapIO

with TemporaryDirectory() as directory:
    storage = DocumentStorage(Path(directory), DocumentPickleFormat(), read_interface=MmapIO())
    with storage.sync_writer() as writer:
        writer.append(0, {"document_id": 1})
    with storage.sync_reader() as reader:
        assert reader.read(0)["document_id"] == 1
```

## Write a custom backend

Subclass `IOReaderAbstract`. Implement `is_supported`, `sync_reader`, and
`async_reader`. The latter two return reader constructors that accept a file path.

| Reader | Required operations |
| --- | --- |
| `IOSyncReaderAbstract` | Constructor, `open`, `close`, `filesize`, and `read(offset, size)`. |
| `IOAsyncReaderAbstract` | Async `open`, `close`, `filesize`, `read`, plus synchronous `close_sync` for cleanup. |

Return exactly the requested bytes as a supported buffer, and raise an error on
short reads. Make concurrent reads independent: avoid shared seek offsets without
synchronization. Buffers must remain valid while their consumers use them. Cleanup
must tolerate partial initialization and repeated calls.

This interface changes how shard bytes are read. `FileSystemStream` and
`FileSystemDocument` still write shards and discover them on disk. See the
[extension reference](../reference/interfaces.md).

## HTTP range requests

An I/O backend can read shard bytes with HTTP range requests. The origin must
return `Content-Length` from `HEAD` and status 206 from a ranged `GET`, with a
body of exactly the requested size. This example uses the optional `aiohttp`
package (`uv pip install aiohttp`).
It implements only async reads and raises from `sync_reader`, so it works with
async storage/dataset readers rather than the synchronous PyTorch loader. The
caller owns the session.

The storage constructor still takes a path. Use a relative path that matches the
object-key prefix, and set that prefix as the session `base_url` with a
trailing slash. aiohttp joins each shard path onto that base. A missing object
raises `FileNotFoundError`, which shard discovery uses to stop. Writing stays
on the filesystem at that path.

```python title="http_io.py"
from functools import partial
from pathlib import PurePath
from typing import NoReturn, final

import aiohttp

from guideloom.types import IOAsyncReaderAbstract, IOReaderAbstract


@final
class HttpIO(IOReaderAbstract):
    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        super().__init__()

    @staticmethod
    def is_supported() -> bool:
        return True

    def sync_reader(self) -> NoReturn:
        raise ValueError("sync reader is not supported")

    def async_reader(self):
        return partial(_HttpIOAsync, session=self._session)


@final
class _HttpIOAsync(IOAsyncReaderAbstract):
    def __init__(self, path: PurePath, *, session: aiohttp.ClientSession) -> None:
        # Relative path, no leading slash, so aiohttp joins it onto session base_url.
        self._url = path.as_posix()
        self._session = session
        self._filesize: int | None = None

    async def open(self) -> None:
        async with self._session.head(self._url) as resp:
            if resp.status == 404:
                raise FileNotFoundError(f"Object {resp.url} does not exist")
            if resp.status >= 400:
                raise RuntimeError(f"HEAD request failed with {resp.status}")
            size = resp.headers.get("Content-Length")
            if size is None:
                raise RuntimeError("Missing Content-Length header")
            self._filesize = int(size)

    async def close(self) -> None:
        pass

    def close_sync(self) -> None:
        pass

    async def filesize(self) -> int:
        if self._filesize is None:
            raise RuntimeError("open was not called")
        return self._filesize

    async def read(self, offset: int, size: int) -> bytearray:
        # One request per read. Multipart ranges can batch simultaneous reads of one object.
        if self._filesize is None:
            raise RuntimeError("open was not called")
        if offset < 0:
            raise IndexError(f"cannot have negative offset, was {offset}")
        if size <= 0:
            return bytearray()

        headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
        async with self._session.get(self._url, headers=headers) as resp:
            if resp.status == 416:
                raise IndexError("requested range not satisfiable")
            if resp.status == 200:
                raise RuntimeError("server ignored the Range header and returned the full object")
            if resp.status != 206:
                raise RuntimeError(f"GET failed with {resp.status}")

            data = await resp.read()

        if len(data) != size:
            raise IndexError(f"read content ({len(data)}B) was shorter than requested size ({size}B)")

        return bytearray(data)
```

Create the session with a `base_url` that ends in `/`, keep it open for the
whole read, and pass `HttpIO` as `read_interface`. Save the implementation as
`http_io.py` and this caller as `read_http.py`. The URL is a placeholder: use
your own trusted dataset host, with a document whose `document_id` is `1`, to run
the assertion:

```python title="read_http.py"
import asyncio
from pathlib import Path

import aiohttp

from http_io import HttpIO
from guideloom import DocumentPickleFormat, DocumentStorage


async def main() -> None:
    async with aiohttp.ClientSession(base_url="https://example.com/data/") as session:
        storage = DocumentStorage(
            Path("datasets/example"),
            DocumentPickleFormat(),
            read_interface=HttpIO(session),
        )
        async with storage.async_reader() as reader:
            assert (await reader.read(0))["document_id"] == 1


if __name__ == "__main__":
    asyncio.run(main())
```
