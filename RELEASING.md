# Releasing GuideLoom

## One-time service setup

1. Enable GitHub Pages in `guidelabs/guideloom` under **Settings → Pages**, with
   **GitHub Actions** as the source. `.github/workflows/release.yml` deploys the
   released documentation to <https://guidelabs.github.io/guideloom/> after PyPI
   publication succeeds. The **Docs check** PR job builds without deploying.
2. Configure the `pypi` and `testpypi` GitHub environments. Restrict production
   publishing to release tags and configure the maintainers who may approve it.
3. On PyPI and TestPyPI, register a Trusted Publisher (a pending publisher for the
   first release): owner `guidelabs`, repository `guideloom`, workflow `release.yml`,
   environment `pypi` or `testpypi` respectively. These are separate services and accounts.
   Confirm the package name is available or controlled by Guide Labs before releasing.

These settings live outside the repository; adding a workflow does not configure them.
See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) and
[GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Prepare a version

1. Update `__version__` in `guideloom/version.py`. Hatchling reads this value for
   distribution metadata; there is no second version to update.
2. Follow [the compatibility policy](docs/compatibility.md). `1.0.0` is
   the first supported release, so do not keep or test pre-1.0 fixtures. After
   it is published, keep those fixtures unchanged, add fixtures for new layouts,
   and document migrations when crossing a breaking boundary.
3. Update `CHANGELOG.md`, replacing the unreleased label with the release date.
   Verify the quick-start and any changed guide examples.
4. Run the checks in `CONTRIBUTING.md`. CI covers Python 3.13 and 3.14 and
   minimum dependencies.

Build into an empty output directory to avoid accidentally uploading older files:

```sh
uv build --out-dir /tmp/guideloom-release
uv run --no-sync twine check --strict /tmp/guideloom-release/*
```

## Rehearse and publish

After merging the release commit, create tag `vVERSION` and publish the
corresponding GitHub release. The release workflow checks the tag against the
package version, runs the repository checks, and builds the archives. It uploads
those archives to TestPyPI, installs the served wheel in a clean environment, and
runs the test suite against that installation. Dependencies come from PyPI. The
same archives are published to PyPI only after those tests pass, and the
documentation is then deployed to GitHub Pages.

A manual run of the **Release** workflow performs the TestPyPI upload and the
installation test, and does not publish to PyPI or GitHub Pages. TestPyPI keeps
every uploaded filename. When the same archives are already present, the upload
is skipped and the tests still run. When a different build of that version is
already present, the workflow stops before PyPI; publish a new version instead of
trying to replace the files.

Publishing a GitHub prerelease can publish a matching Python prerelease and its
documentation; choose the version deliberately.

Confirm the PyPI README and project links render correctly, then verify installation
in a fresh environment. For a bad release, yank it on PyPI and publish a corrected
version; do not attempt to replace an existing archive.
