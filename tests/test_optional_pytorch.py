import sys
import subprocess


def test_package_imports_without_pytorch() -> None:
    code = """
import sys
sys.modules["torch"] = None
sys.modules["torchdata"] = None
import guideloom
assert "StatefulDataLoader" not in vars(guideloom)
for statement in (
    "from guideloom import StatefulDataLoader",
    "from guideloom.pytorch import StatefulDataLoader",
):
    try:
        exec(statement, {"guideloom": guideloom})
    except ModuleNotFoundError as error:
        if error.name not in {"torch", "torchdata"}:
            raise SystemExit(f"{statement}: {error}") from error
    else:
        raise SystemExit(f"imported without torch: {statement}")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
