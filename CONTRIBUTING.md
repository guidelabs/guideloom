# Contributing

Use Python 3.13, 3.14, or 3.15, Node.js 24, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Create an isolated environment and install development dependencies:

```sh
uv venv --python 3.13
uv pip install -e . --group ci --group docs --group release
npm --prefix docs/components ci
```

Run the checks used by CI:

```sh
uv run --no-sync ruff check guideloom tests
uv run --no-sync ruff format --check guideloom tests
uv run --no-sync basedpyright
uv run --no-sync pytest
uv run --no-sync npm --prefix docs/components run build
```

CI also builds the wheel and source distribution and checks their metadata.
See [RELEASING.md](RELEASING.md) for the release procedure.

## Documentation

The site uses [Zensical](https://zensical.org/) and
[mkdocstrings-python](https://mkdocstrings.github.io/python/). The latter parses
source files, so the documentation can build without NumPy, Torch, or the other
runtime dependencies. Use Node.js 24 for the interactive components:

```sh
uv run --no-sync npm --prefix docs/components run serve
```

Open [http://localhost:8000/](http://localhost:8000/) in your browser. The server
rebuilds and reloads as you edit; press `Ctrl+C` to stop it. For setup without
runtime dependencies, see [View documentation locally](README.md#view-documentation-locally).

Edit `zensical.toml` to change navigation. Tutorials and guides live under `docs/`;
API pages explicitly select public objects and methods. Use Google-style docstrings.

The storage-format comparison is a Svelte component in `docs/components/`.
That source directory is excluded from the published site.
The build command compiles it before building the site. The preview command also
watches the component and rebuilds it when its source changes. Generated JavaScript
and third-party notices go into `docs/assets/generated/`, which is ignored by Git.
Commit the Svelte source, build scripts, and npm lockfile.

Run `uv run --no-sync npm --prefix docs/components run build` before submitting.
It runs Zensical in strict mode. The documentation example check
executes the Python code blocks in the published pages. Make each block runnable
on CPU, use temporary directories, and avoid external datasets or network access.

`uv pip install` installs the selected groups into `.venv`. `uv run --no-sync` uses
that environment without resolving or changing it, including when CI tests the
minimum dependency versions. `uv build` builds the wheel and source distribution
through the Hatchling backend.

## Compatibility changes

Follow [the compatibility policy](docs/compatibility.md). `1.0.0` is the
first supported release; do not add a compatibility fixture for `0.1.0`. After
`1.0.0` is published, keep historical fixtures in `tests/fixtures/compatibility/`
unchanged. Add fixtures for new releases rather than regenerating older ones to
make a test pass. Explain public behavior changes and migrations in `CHANGELOG.md`.
