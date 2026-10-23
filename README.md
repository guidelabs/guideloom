# GuideLoom

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/guidelabs/guideloom/main/docs/assets/guideloom-logo-dark.svg">
    <img alt="GuideLoom" src="https://raw.githubusercontent.com/guidelabs/guideloom/main/docs/assets/guideloom-logo.svg" width="420">
  </picture>
</p>

**Flexible data. Reliable training. Built to scale.**

GuideLoom is dataset storage and a resumable PyTorch DataLoader for large-scale
training. Keep documents, token streams, and span annotations in one data pipeline,
with global shuffling, independent distributed reads, and exact checkpointing.

[Quick-start](https://guidelabs.github.io/guideloom/quick-start/) ·
[Documentation](https://guidelabs.github.io/guideloom/) ·
[Design story](https://www.guidelabs.ai/post/dataloader-how-we-built/#our-functional-and-performance-requirements)

## Why GuideLoom?

| Benefit | What you get |
| --- | --- |
| **Support complex data layouts** | Whole documents, fixed-length streams, or streams with variable-size span annotations. Keep concept lists and document IDs connected to the training windows they describe. |
| **Your schema, your context window** | Custom encoders and decoders for documents and metadata; multiple aligned stream fields for tokens, loss masks, or per-token concepts. Select stream window sizes at read time. |
| **Instant global shuffling** | Seeded sampling without replacement across all shards, without allocating a dataset-sized permutation or storing a shuffled dataset copy for each epoch. |
| **Exact resumption with prefetching** | Checkpoint the position delivered to training while workers prepare future batches. Resume at the next batch with the same dataset, seed, and batch partitioning. |
| **Independent distributed reads** | Each rank reads its own samples from local or shared storage. No communication between loaders or whole-shard staging on every node. Designed for HPC parallel filesystems, with extensible read backends. |
| **High performance from the ground up** | NumPy buffer views for compatible encodings, GIL-free compiled indexing and shuffling, plus configurable workers and background prefetching to overlap data preparation with training. |

See [performance tips](https://guidelabs.github.io/guideloom/advanced/performance/)
for encoding and I/O tuning, and
[save and resume](https://guidelabs.github.io/guideloom/intermediate/save-resume/)
for checkpointing examples and the settings needed to preserve sample order.

Requires Python 3.13, 3.14, or 3.15. See the
[compatibility policy](https://guidelabs.github.io/guideloom/compatibility/).

## Install

```sh
uv venv --python 3.15
uv pip install "guideloom[torch]"
```

The `torch` extra installs PyTorch for `StatefulDataLoader`.
Dataset storage and readers are available from the base package:
`uv pip install guideloom`.

For the unreleased version, install from a checkout with `uv pip install -e ".[torch]"`.

## Learn GuideLoom

Start with the [quick-start](https://guidelabs.github.io/guideloom/quick-start/):
write a dataset once in a preprocessing script, then open it later in a training script.
The [documentation](https://guidelabs.github.io/guideloom/) follows that workflow:

- **Beginner:** choose a storage format, write independent shards in parallel, and load PyTorch batches from storage across ranks.
- **Intermediate:** save and resume, learn NumPy structured arrays, and define encodings for each storage format.
- **Advanced:** tune performance, use dataset and async readers directly, implement I/O backends, and integrate another framework such as JAX.

## View documentation locally

From a repository checkout, use [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.13, and Node.js 24. Run these setup commands from the repository root:

```sh
uv venv --python 3.13
uv pip install --group docs
uv run --no-sync npm --prefix docs/components ci
```

Start the local preview server:

```sh
uv run --no-sync npm --prefix docs/components run serve
```

Open [http://localhost:8000/](http://localhost:8000/) in your browser. Keep the
server running while you edit files under `docs/`; it rebuilds the site and
reloads the browser automatically, including changes to the interactive components.
Press `Ctrl+C` in the terminal to stop it.

The documentation tools do not require GuideLoom's runtime dependencies.

## License

GuideLoom is licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) and [licenses/](licenses/) for third-party attributions and license terms.
