"""Install the TestPyPI release in a clean environment and run its tests.

The release workflow uploads the built archives, then uses this script to install
the served wheel and run the repository tests before publishing to PyPI.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import email
import hashlib
import json
import os
import runpy
import shutil
import subprocess
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
POLL_SECONDS = 10 * 60
POLL_INTERVAL_SECONDS = 15


class TransientReleaseError(Exception):
    """TestPyPI could not be read, and the request can be repeated."""


def run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


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


def wait_for_release(version: str, local: dict[str, str]) -> dict[str, tuple[str, str]]:
    deadline = time.monotonic() + POLL_SECONDS
    while True:
        try:
            payload = fetch_release(version)
        except TransientReleaseError as error:
            print(f"Waiting for {PROJECT} {version} on TestPyPI ({error})", file=sys.stderr)
            if time.monotonic() >= deadline:
                raise SystemExit(f"Timed out waiting for {PROJECT} {version} on TestPyPI") from error
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        remote = None if payload is None else remote_files(payload)
        state = release_state(local, remote)
        if state == "present" and remote is not None:
            return remote
        if state == "different":
            raise SystemExit(mismatch_message(version, state))
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for {PROJECT} {version} on TestPyPI")
        print(f"Waiting for {PROJECT} {version} on TestPyPI ({state})", file=sys.stderr)
        time.sleep(POLL_INTERVAL_SECONDS)


def download(url: str, destination: Path) -> None:
    deadline = time.monotonic() + POLL_SECONDS
    while True:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except (urllib.error.URLError, TimeoutError) as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Could not download {url}") from error
            time.sleep(POLL_INTERVAL_SECONDS)


def command_test(dist: Path) -> None:
    version, local = local_release(dist)
    remote = wait_for_release(version, local)
    with TemporaryDirectory(prefix="guideloom-testpypi-") as directory:
        temporary = Path(directory)
        for name, digest in local.items():
            download(remote[name][1], temporary / name)
            if file_digest(temporary / name) != digest:
                raise SystemExit(f"Downloaded {name} does not match the built archive")
        wheel = next(temporary.glob("*.whl"))
        environment = temporary / "venv"
        run("uv", "venv", str(environment), "--python", sys.executable, cwd=temporary)
        python = environment / "bin" / "python"
        run("uv", "pip", "install", "--python", str(python), str(wheel), cwd=temporary)
        smoke = (
            "import os; from pathlib import Path; from importlib.metadata import distribution; "
            "import importlib; "
            "import guideloom; "
            "from guideloom.storage import StreamStorage, DocumentStorage, StreamWithMetaStorage; "
            "d = distribution('guideloom'); "
            "assert d.version == guideloom.__version__ == os.environ['EXPECTED_VERSION']; "
            "assert Path(guideloom.__file__).is_relative_to(Path(__import__('sys').prefix)); "
            "assert Path(guideloom.__file__).with_name('py.typed').is_file(); "
            "assert not hasattr(guideloom, 'ABI_VERSION'); "
            "assert 'torch' in (d.metadata.get_all('Provides-Extra') or []); "
            "required = [req for req in (d.requires or []) if 'extra ==' not in req]; "
            "names = {req.split(';')[0].split('[')[0].split('>')[0].split('<')[0].split('=')[0]"
            ".split('!')[0].split('~')[0].strip() for req in required}; "
            "assert 'torch' not in names and 'torchdata' not in names; "
            "try:\n"
            "    importlib.import_module('guideloom.pytorch')\n"
            "except ModuleNotFoundError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('guideloom.pytorch imported without the torch extra')"
        )
        environment_variables = os.environ.copy()
        environment_variables["EXPECTED_VERSION"] = version
        subprocess.run(
            [str(python), "-I", "-c", smoke],
            cwd=temporary,
            check=True,
            env=environment_variables,
        )
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--group",
            f"{ROOT / 'pyproject.toml'}:test",
            cwd=temporary,
        )
        run(
            str(python),
            "-I",
            "-m",
            "pytest",
            str(ROOT / "tests"),
            "-c",
            str(ROOT / "pyproject.toml"),
            "-q",
            "--tb=short",
            cwd=temporary,
        )
    print("TestPyPI package checks passed")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "test"):
        command = subparsers.add_parser(name)
        command.add_argument("--dist", type=Path, required=True)
    arguments = parser.parse_args()
    dist = arguments.dist.resolve()
    if arguments.command == "status":
        command_status(dist)
    else:
        command_test(dist)


if __name__ == "__main__":
    main()
