"""The paper's reproduction scripts parse their options and plan every stage."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"
SCRIPTS = sorted(PAPER_DIR.glob("reproduce_*.sh"))

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_every_paper_script_is_present():
    """The four scripts of paper/README.md ship with the repository."""
    assert [script.name for script in SCRIPTS] == [
        "reproduce_ablations.sh",
        "reproduce_experiment1_manylabs.sh",
        "reproduce_experiment2_twins.sh",
        "reproduce_experiment3_bayesian.sh",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_dry_run_plans_every_stage(script, tmp_path):
    """A dry run prints the plan and commands of all stages and runs nothing."""
    result = subprocess.run(
        [
            "bash", str(script), "--dry-run", "--gpus", "1", "--cpus", "8",
            "--memory-gb", "64", "--python", sys.executable,
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "(dry run: the commands below are printed, not run)" in result.stdout
    assert result.stdout.count("\n+ ") > 10
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_help_prints_the_header(script):
    """--help prints the script's documentation."""
    result = subprocess.run(
        ["bash", str(script), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "STAGES" in result.stdout and "REPRODUCIBILITY" in result.stdout
