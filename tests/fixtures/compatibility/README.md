# Compatibility fixtures

`1.0.0` is the first supported release. Do not add a `0.1.0` directory or test
that current code can read datasets or checkpoints produced before `1.0.0`.

After `1.0.0` is published, add a `1.0.0/` directory for layouts that later
releases in the same major version must keep readable. Leave a published fixture
unchanged. Add a new directory for a later release instead of regenerating an
old one to hide a compatibility failure.
