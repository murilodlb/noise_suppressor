import subprocess
import sys
from pathlib import Path


def test_no_torch_nn_alias_in_codebase():
    """Asserts that no python module in src/ or tests/ imports torch.nn under the 'nn' alias."""
    repo_root = Path(__file__).resolve().parent.parent
    python_files = sorted(
        list((repo_root / "src").glob("**/*.py")) + list((repo_root / "tests").glob("**/*.py"))
    )

    forbidden_patterns = [
        "import torch.nn as nn",
        "from torch import nn",
    ]

    violations = []
    current_file = Path(__file__).resolve()
    for file_path in python_files:
        if file_path.resolve() == current_file:
            continue
        content = file_path.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), start=1):
            for pattern in forbidden_patterns:
                if pattern in line:
                    violations.append(
                        f"{file_path.relative_to(repo_root)}:{line_num}: '{line.strip()}'"
                    )

    assert not violations, (
        "Found forbidden 'nn' alias imports (use full 'torch.nn' instead):\n"
        + "\n".join(violations)
    )


def test_ruff_lint_clean():
    """Executes 'ruff check .' and asserts zero lint or style violations."""
    cmd = [sys.executable, "-m", "ruff", "check", "."]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"ruff check failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_ruff_format_clean():
    """Executes 'ruff format --check .' and asserts code is properly formatted."""
    cmd = [sys.executable, "-m", "ruff", "format", "--check", "."]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"ruff format --check failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_pyright_type_checking_clean():
    """Executes pyright and asserts zero static type errors across the codebase."""
    cmd = [sys.executable, "-m", "pyright"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"pyright failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
