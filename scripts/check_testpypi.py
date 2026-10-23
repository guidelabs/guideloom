"""Check whether release archives are missing or already published on TestPyPI.

The release workflow uses this check before uploading so it can skip identical
archives and reject versions whose published files differ from the build.
"""

from pathlib import Path
import email
import hashlib
import json
import os
import runpy
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "guideloom"
INDEX = "https://test.pypi.org/pypi"
FILE_HOSTS = {"test.pypi.org", "test-files.pythonhosted.org"}
USER_AGENT = "guideloom-release (+https://github.com/guidelabs/guideloom)"
TRANSIENT_HTTP = {429, 500, 502, 503, 504}
POLL_INTERVAL_SECONDS = 15


class TransientReleaseError(Exception):
    """TestPyPI could not be read, and the request can be repeated."""


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        version = email.message_from_bytes(wheel.read(metadata_name))["Version"]
    if not version:
        raise SystemExit(f"{path.name} has no version metadata")
    return version


def local_release(dist: Path) -> tuple[str, dict[str, str]]:
    archives = sorted(path for path in dist.iterdir() if path.is_file() and not path.name.startswith("."))
    wheels = [path for path in archives if path.suffix == ".whl"]
    sources = [path for path in archives if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sources) != 1 or len(archives) != 2:
        raise SystemExit("Expected exactly one wheel and one source distribution in the dist directory")
    version = wheel_version(wheels[0])
    source_version = runpy.run_path(str(ROOT / "guideloom" / "version.py"))["__version__"]
    if version != source_version:
        raise SystemExit(
            f"Built wheel version {version} does not match guideloom/version.py ({source_version})"
        )
    return version, {path.name: file_digest(path) for path in archives}


def fetch_release(version: str) -> dict[str, object] | None:
    version_path = urllib.parse.quote(version)
    request = urllib.request.Request(
        f"{INDEX}/{PROJECT}/{version_path}/json",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        if error.code in TRANSIENT_HTTP:
            raise TransientReleaseError(str(error.code)) from error
        raise
    except (urllib.error.URLError, TimeoutError) as error:
        raise TransientReleaseError(str(error)) from error
    if not isinstance(payload, dict):
        raise SystemExit("TestPyPI returned an unexpected release document")
    return payload


def remote_files(payload: dict[str, object]) -> dict[str, tuple[str, str]]:
    info = payload.get("info")
    if not isinstance(info, dict):
        raise SystemExit("TestPyPI release document has no info object")
    if info.get("yanked"):
        raise SystemExit("This version is yanked on TestPyPI; publish a new version")
    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise SystemExit("TestPyPI release document has no file list")
    files: dict[str, tuple[str, str]] = {}
    for item in urls:
        if not isinstance(item, dict):
            raise SystemExit("TestPyPI release document has an invalid file entry")
        filename = item.get("filename")
        digests = item.get("digests")
        url = item.get("url")
        if not isinstance(filename, str) or not isinstance(url, str) or not isinstance(digests, dict):
            raise SystemExit("TestPyPI release document is missing file details")
        digest = digests.get("sha256")
        if item.get("yanked") or not isinstance(digest, str):
            raise SystemExit(f"{filename} is not a usable TestPyPI file")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in FILE_HOSTS:
            raise SystemExit(f"Unexpected TestPyPI file URL: {url}")
        files[filename] = (digest, url)
    return files


def release_state(local: dict[str, str], remote: dict[str, tuple[str, str]] | None) -> str:
    if remote is None:
        return "missing"
    published = {
        name: value for name, value in remote.items() if name.endswith(".whl") or name.endswith(".tar.gz")
    }
    if set(published) < set(local):
        return "incomplete"
    if set(published) != set(local):
        return "different"
    if any(published[name][0] != digest for name, digest in local.items()):
        return "different"
    return "present"


def write_state(state: str) -> None:
    print(f"TestPyPI state: {state}", file=sys.stderr)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"state={state}\n")


def mismatch_message(version: str, state: str) -> str:
    if state == "different":
        return (
            f"TestPyPI already has a different {PROJECT} {version} release. "
            "Uploaded filenames cannot be replaced; publish a new version."
        )
    return f"TestPyPI has an incomplete {PROJECT} {version} release. Publish a new version."


def command_status(dist: Path) -> None:
    version, local = local_release(dist)
    attempts = 3
    payload = None
    for attempt in range(1, attempts + 1):
        try:
            payload = fetch_release(version)
            break
        except TransientReleaseError as error:
            if attempt == attempts:
                raise SystemExit(f"Could not read {PROJECT} {version} from TestPyPI") from error
            time.sleep(POLL_INTERVAL_SECONDS)
    remote = None if payload is None else remote_files(payload)
    state = release_state(local, remote)
    if state in {"different", "incomplete"}:
        raise SystemExit(mismatch_message(version, state))
    write_state(state)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("status")
    command.add_argument("--dist", type=Path, required=True)
    arguments = parser.parse_args()
    command_status(arguments.dist.resolve())


if __name__ == "__main__":
    main()
