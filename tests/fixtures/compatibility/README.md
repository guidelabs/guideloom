# Compatibility fixtures

These datasets were written with `v1.0.0` using the storage formats from
`tests/test_storage_*.py`. Paths omit the release version; keep the files
unchanged until support ends in the next major release.

`tests/test_compatibility.py` defines small deterministic datasets (with seeded
NumPy RNGs for stream data). Its writer and reader tests use the same generators.
Tests read the checked-in files without rewriting them.

To write fixtures into a new directory using the installed GuideLoom version:

```sh
python -m tests.test_compatibility /tmp/compatibility-fixtures
```
