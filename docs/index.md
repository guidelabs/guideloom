# GuideLoom {: hidden }

<p align="center">
  <img alt="GuideLoom" src="assets/guideloom-logo.svg#only-light" width="420">
  <img alt="GuideLoom" src="assets/guideloom-logo-dark.svg#only-dark" width="420">
</p>

## Flexible data. Reliable training. Built to scale.

GuideLoom is randomly-addressable dataset storage and a resumable PyTorch DataLoader for large-scale distributed training. Keep documents, token streams, and span annotations in one data pipeline,
with global shuffling, distributed reads, exact checkpointing, and high throughput.

[Quick start](quick-start.md){ .md-button .md-button--primary }
[Tutorial](#beginner){ .md-button }
[Blogpost](https://www.guidelabs.ai/post/dataloader-how-we-built/){ .md-button }

## Flexible data formats

Choose a layout to match your training samples, with custom encoders and decoders
for your data:

- **Streams:** Concatenate tokens or other fixed-size values into a continuous
  stream. Read fixed-length windows, with optional aligned fields such as loss masks.
- **Documents:** Keep variable-length records, such as conversations, separate.
  Each sample is a complete record; your collator handles padding.
- **Streams with metadata:** Attach variable-size annotations to spans in a stream,
  such as concept lists or document IDs. Each window includes the relevant metadata,
  even when it crosses document boundaries.

Switch formats below to compare how the same documents become training samples.
Change the window size or select a token to explore.

<div data-storage-formats>
  <p>The interactive comparison requires JavaScript.</p>
</div>

See [choose a storage format](beginner/storage.md) for a comparison table and examples.

## Why GuideLoom?

<div class="grid cards" markdown>

-   :lucide-layers: **Support complex data layouts**

    Train on complete documents, fixed-length streams, or streams with annotations
    on token spans. Carry document IDs for traceability or variable-size concept
    lists into each training window, even across document boundaries.

    [Explore the storage formats](beginner/storage.md)

-   :lucide-sliders-horizontal: **Choose what your data contains**

    Define your own encoders and decoders for documents and span metadata. Store
    multiple aligned fields in a stream, such as token IDs, loss masks, and
    per-token concepts. Change the context window at read time without rewriting
    the stream.

    [Define a stream with metadata](intermediate/stream-metadata.md)

-   :lucide-shuffle: **Instant global shuffling**

    Seeded shuffling samples without replacement across the entire dataset,
    including shard boundaries. Compute shuffled indices as you need them,
    without a dataset-sized permutation or a separate shuffled copy for each
    epoch.

    [Load shuffled batches](beginner/load-batches.md)

-   :lucide-bookmark-check: **Resume exactly, with prefetching**

    Prepare future batches in the background while checkpointing the position
    actually delivered to training. With the same dataset, seed, and batch
    partitioning, resume at the next batch without skipping prefetched data
    or repeating consumed batches.

    [Save and resume training](intermediate/save-resume.md)

-   :lucide-network: **Scale reads across GPUs and nodes**

    Each rank reads its assigned samples from shared storage, without
    communication between loaders or staging whole shards on every node.
    Use local disks or HPC parallel filesystems; extend the read backend for
    other storage systems.

    [Explore I/O backends](advanced/io-backends.md)

-   :lucide-zap: **High performance from the ground up**

    NumPy buffer views avoid extra decoding copies for compatible numeric
    formats. Compiled indexing and shuffling release the GIL, while configurable
    workers and prefetching overlap CPU data preparation with training.

    [Tune the read pipeline](advanced/performance.md)

</div>

## Start with a complete example

The [quick-start](quick-start.md) has two scripts: one writes a persistent dataset;
the other opens it later and produces PyTorch batches. Preprocessing and training
can run at different times and on different machines with access to the same files.

## Beginner

Follow these in order. You only need basic Python and NumPy arrays; structured
arrays are introduced when you need them.

1. [Choose a storage format](beginner/storage.md): documents, streams, or streams with metadata.
2. [Write a dataset](beginner/write-dataset.md): prepare shards, including parallel writers with exclusive shard ownership.
3. [Read into PyTorch](beginner/load-batches.md): create a PyTorch dataloader, collate samples, and partition loading by rank and world size.

## Intermediate

- [Save and resume](intermediate/save-resume.md): checkpoint state and requirements for preserving sample order.
- [NumPy structured arrays](intermediate/numpy-structured-arrays.md): are central to the design of guideloom, so while it's a NumPy API we include a breif tutorial.
- Understand each storage type in detail with examples:
  - [Document storage](intermediate/documents.md) - for variable-length records that should be kept seperate.
  - [streams](intermediate/streams.md) for concatenating records, such as token-ids in pre-training.
  - [streams with metadata](intermediate/stream-metadata.md) for attaching custom metadata,
    to spans of stream content.

## Advanced

- [Performance tips](advanced/performance.md): pread, workers, prefetching, serialization, and device placement.
- [Use the dataset API directly](advanced/dataset-api.md).
- [Use asynchronous readers](advanced/async-api.md).
- [Write a custom I/O backend](advanced/io-backends.md), including HTTP range reads.
- [Use another framework](advanced/other-frameworks.md), such as JAX.

The [API reference](reference/index.md) documents storage, built-in formats,
I/O backends, datasets, and extension interfaces.
Read the [compatibility policy](compatibility.md) before upgrading an
existing dataset or training job.
