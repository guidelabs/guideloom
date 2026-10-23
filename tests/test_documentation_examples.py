from pathlib import Path
from typing import Any
import os
import re
import sys
import subprocess

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _examples() -> list[Any]:
    root = DOCS
    examples = []
    for page in sorted(root.rglob("*.md")):
        if page.is_relative_to(root / "components"):
            continue
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r"^```(?:python|py)\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
            line = text.count("\n", 0, match.start()) + 2
            examples.append(pytest.param(match.group(1), id=f"{page.relative_to(root)}:{line}"))
    return examples


@pytest.mark.parametrize("code", _examples())
def test_documented_example_runs_on_cpu(code: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _script(page: str, filename: str) -> str:
    match = re.search(
        rf'^```python title="{re.escape(filename)}"\n(.*?)^```',
        (DOCS / page).read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing {filename} in {page}"
    return match.group(1)


def _run_script(path: Path, **env: str) -> str:
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=path.parent,
        env={**os.environ, "RANK": "0", "WORLD_SIZE": "1", **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_quick_start_writes_then_reads_in_separate_processes(tmp_path: Path) -> None:
    shared_format = tmp_path / "dataset_format.py"
    shared_format.write_text(_script("quick-start.md", shared_format.name))
    for filename in ("write_dataset.py", "train.py"):
        script = tmp_path / filename
        script.write_text(_script("quick-start.md", filename))
        _run_script(script)
    # The dataset survives the writer process and can be reopened repeatedly.
    _run_script(tmp_path / "train.py")


def test_parallel_shards_are_read_once_across_ranks(tmp_path: Path) -> None:
    writer = tmp_path / "write_shards.py"
    writer.write_text(_script("beginner/write-dataset.md", writer.name))
    _run_script(writer)
    reader = tmp_path / "train_distributed.py"
    reader.write_text(_script("beginner/load-batches.md", reader.name))
    per_rank = []
    for rank in range(2):
        output = _run_script(reader, RANK=str(rank), WORLD_SIZE="2")
        tokens = [int(value) for line in output.splitlines() for value in re.findall(r"\d+", line)[1:]]
        assert len(tokens) == 32
        per_rank.append(tokens)
    assert set(per_rank[0]).isdisjoint(per_rank[1])
    assert sorted(per_rank[0] + per_rank[1]) == list(range(64))


@pytest.mark.parametrize(
    ("page", "filename"),
    [
        ("advanced/io-backends.md", "http_io.py"),
        ("advanced/io-backends.md", "read_http.py"),
        ("advanced/other-frameworks.md", "train_jax.py"),
    ],
)
def test_optional_example_compiles(page: str, filename: str) -> None:
    # These examples need optional dependencies or an application-owned HTTP host.
    compile(_script(page, filename), filename, "exec")


def test_jax_example_runs_when_installed(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    script = tmp_path / "train_jax.py"
    script.write_text(_script("advanced/other-frameworks.md", script.name))
    _run_script(script)
