# Compatibility policy

GuideLoom follows semantic versioning, starting with `1.0.0`. Check your version
with `guideloom.__version__`.

| Release | Contract |
| --- | --- |
| Patch or minor | Preserve public APIs, dataset readability, checkpoint resumption, and sample ordering. |
| Major | May introduce breaking changes, documented with migration steps. |
| Prerelease | Datasets and checkpoints are supported only by the exact version that produced them. |

## Public API

The API reference and documented imports define the supported API, including
extension interfaces. Private implementations are internal. APIs are deprecated
in a minor release before removal in a major release.

## Stored datasets

Newer releases read datasets from older releases with the same major version.
Older readers may not read newer files.

Dataset files do not record or validate the producing package version. Track it
with your dataset by for example including it in the filepath. Preserve any custom codecs, schemas, and I/O implementations. Your application owns their compatibility.

## Checkpoints and sample order

Within a major version, newer releases resume older checkpoints with the same
remaining global sample order. Keep the dataset, dataset view, and shuffle/drop
settings unchanged. Use an explicit shuffle seed. Batch size and world size may
change on resumption; samples are repartitioned across the new ranks and batches.
With `drop_last=True`, the new global batch size determines the dropped tail.

Checkpoints do not validate all these settings or record the package version.
Your application must preserve them. This guarantees sample order, not identical
tensor values; see [save and resume](intermediate/save-resume.md#preserve-sample-order).

## Supported environments

CPython 3.13–3.15 is supported with the dependency versions allowed by
`pyproject.toml`, including the `torch` extra. Free-threaded Python is unsupported.
CUDA requires a compatible PyTorch/CUDA installation; CPU CI does not validate GPU behavior.

Dropping a supported Python version requires a major release. Dependency changes
must preserve the guarantees above.
