# Choose a storage format

Choose based on what one training sample should contain. A **document** is one
record you want to keep separate, such as a conversation. A **stream** joins
records end to end, then reads fixed-length windows, such as a language model's
context window. Metadata can describe a document or a smaller span within it.

| Your data | Choose | How a sample is read |
| --- | --- | --- |
| Separate conversations or variable-length records | `DocumentStorage` | One whole record. Omit `sequence_length`. |
| Concatenated tokens, optionally with a fixed-size value per token | `StreamStorage` | A fixed window. Pass `sequence_length`. |
| Concatenated tokens with variable-size annotations on spans | `StreamWithMetaStorage` | A fixed window. Pass `sequence_length`. |

Training reads through `StatefulDataLoader.from_storage`, which wraps the storage
in the matching dataset view. Open `StreamDataset` or `DocumentDataset` yourself
only for indexed access outside the PyTorch loader; see
[the dataset API](../advanced/dataset-api.md).

## Compare the formats interactively

Switch modes to see how the same two documents become training samples. Change
the window size or select a token to explore stream windows, including windows
that cross document boundaries. In document mode, each sample is a whole document.

<div data-storage-formats>
  <p>The interactive comparison requires JavaScript.</p>
</div>

This illustration treats words as tokens and adds an explicit `<eos>` token to
each stream document. Your preprocessing chooses the real tokenizer and any
separator tokens; GuideLoom does not insert them automatically. Concepts here
cover whole documents, but stream metadata can also describe smaller spans.

## Documents

Use documents when record boundaries matter. Each read returns one whole decoded
record. GuideLoom does not join documents or pad them; your PyTorch collator can
pad a batch to a common length.

A format controls how each record becomes bytes. `DocumentPickleFormat` is a
convenient starting point for trusted data. For a compact application format,
use JSON, MessagePack, or binary arrays through `DocumentStorageFormat`.
See the [document example](../intermediate/documents.md).

## Streams

Use a stream when training should fill every context window. Appending two
records with token IDs `[10, 11, 12]` and `[13, 14, 15]` creates the logical
stream `[10, 11, 12, 13, 14, 15]`. A four-token window can cross the original
record boundary.

The built-in `StreamNumpyFormat` takes a dictionary of one-dimensional NumPy
arrays. Start with a single `tokens` array, as in the [quick-start](../quick-start.md).
You can add fields such as a document ID or a loss mask if every token has a
fixed-size value for that field. You do not need to know structured arrays to
start; the [introduction](../intermediate/numpy-structured-arrays.md) explains the underlying
representation. See the [stream example](../intermediate/streams.md).

`sequence_length` selects complete, non-overlapping windows. A trailing partial
window is omitted. The window size is a read-time choice, independent of how you
split records during preprocessing.

## Streams with metadata

Use a stream with metadata when an annotation varies in size, such as a list of
concepts for a paragraph. Store the token array together with a mapping from each
token to an annotation. Adjacent tokens can share one stored annotation, and a
read window can include parts of several annotated spans.

The token array is called **stream data**. The separately encoded annotations
are **document data**. Your `StreamWithMetaStorageFormat` encodes, decodes,
and aligns them. See the [stream-with-metadata example](../intermediate/stream-metadata.md).
For a single fixed-size document ID per token, a plain stream field is simpler.

## Reopen a dataset

These three storage layouts use different files and are not interchangeable
readers. Reopen a dataset with its original storage class and serialization
format. Preserve that schema with the data; it is not inferred from file contents.

Continue with [writing a dataset](write-dataset.md).
