# NumPy structured arrays

A normal NumPy array has one value type, such as a 32-bit token ID. A structured
array has named fields in each element, like a table row with a fixed binary
layout. Each field can have a different type, but every row takes the same number
of bytes.

See NumPy's [structured arrays guide](https://numpy.org/doc/stable/user/basics.rec.html)
for details on defining fields, accessing values, and controlling memory layout.

## Define and access fields

```python
import numpy as np

row_dtype = np.dtype([("tokens", "<i4"), ("document_ids", "<u4"), ("loss_mask", "u1")])
rows = np.array([(10, 7, 1), (11, 7, 0), (12, 8, 1)], dtype=row_dtype)

assert rows.ndim == 1
assert rows.shape == (3,)
assert row_dtype.itemsize == 9
assert rows[0]["tokens"] == 10
assert rows["document_ids"].tolist() == [7, 7, 8]

encoded = rows.tobytes()
restored = np.frombuffer(encoded, dtype=row_dtype)
assert np.array_equal(restored, rows)
```

`<i4` is a little-endian signed four-byte integer, `<u4` is an unsigned four-byte
integer, and `u1` is an unsigned byte. With the default packed layout, these fields
take nine bytes per row. `rows` is one-dimensional even though it has three
fields. Its token field is an array view with one token every nine bytes.

`np.frombuffer` interprets existing bytes using a dtype without copying them.
Keep the backing buffer alive while using its views, and use `.copy()` when you
need independent writable memory. Bytes-backed arrays are read-only.

## Use fields with GuideLoom

`StreamNumpyFormat` packs a dictionary of equally sized one-dimensional arrays
into a structured array. On reading, it returns a dictionary of field views.
You can use ordinary arrays in your preprocessing and collator; the format
handles the structured representation. See the [stream example](streams.md).

A custom `StreamStorageFormat` may instead return a structured array directly.
Choose plain stream fields for small, fixed-size values such as masks or IDs.
Use [stream metadata](stream-metadata.md) for variable-length payloads
such as lists of concepts.

## Preserve the binary layout

Keep field names, order, types, and byte order consistent between writing and
reading. Filesystem stream storage normalizes dtypes to little endian; using
explicit little-endian types makes your schema clear. Application payloads in
`DocumentStorage` or document metadata use the encoding you define.

Do not use `object` fields for raw storage: they hold Python object references,
not a portable serialization of those objects. Field views can also be strided;
a tensor conversion or collator may need a contiguous copy. `np.stack` creates
a new batch array, so a buffer-based storage read does not make the whole
training pipeline zero-copy.
