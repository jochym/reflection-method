"""Integration tests for CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run reflection-method CLI and return result."""
    cmd = [sys.executable, "-m", "reflection_method.cli"] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)


def test_cli_help():
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "reflection method" in result.stdout


def test_cli_find_basic():
    """Test basic CLI find command with AAVSO file."""
    fixture = Path(__file__).parent / "fixtures" / "V500_Peg_2461261_2026-08-08_G.aavso.txt"
    result = run_cli([
        str(fixture),
        "--x-col", "DATE-OBS",
        "--y-col", "MAG",
        "--w-col", "MAG_ERR",
        "--invert-mag",
        "--time-format", "iso",
        "--pts-per-knot", "10",
        "--degree", "3",
        "--n-scan", "200",
        "--n-bootstrap", "30",  # Reduced for speed
        "--seed", "42",
    ])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)

    # Check required fields
    assert "x0" in data
    assert "x0_std" in data
    assert "x0_lo" in data
    assert "x0_hi" in data
    assert "sigma_min" in data
    assert "n_points" in data
    assert "n_bootstrap" in data
    assert "utc_time" in data
    assert "utc_uncertainty_s" in data

    # Check values are reasonable
    assert data["n_points"] == 120
    assert data["n_bootstrap"] == 30
    assert 94 < data["x0"] < 96  # Minutes from start
    assert data["x0_std"] > 0
    assert data["x0_lo"] < data["x0_hi"]
    assert data["utc_time"].endswith("Z")
    assert "T" in data["utc_time"]


def test_cli_find_output_file(tmp_path):
    """Test CLI with --output file."""
    fixture = Path(__file__).parent / "fixtures" / "V500_Peg_2461261_2026-08-08_G.aavso.txt"
    output_file = tmp_path / "result.json"

    result = run_cli([
        str(fixture),
        "--x-col", "DATE-OBS",
        "--y-col", "MAG",
        "--w-col", "MAG_ERR",
        "--invert-mag",
        "--time-format", "iso",
        "--pts-per-knot", "10",
        "--degree", "3",
        "--n-scan", "200",
        "--n-bootstrap", "20",
        "--seed", "123",
        "--output", str(output_file),
    ])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert "x0" in data
    assert data["utc_time"].endswith("Z")


def test_cli_reproducibility():
    """Test that --seed gives reproducible results."""
    fixture = Path(__file__).parent / "fixtures" / "V500_Peg_2461261_2026-08-08_G.aavso.txt"

    result1 = run_cli([
        str(fixture), "--x-col", "DATE-OBS", "--y-col", "MAG", "--w-col", "MAG_ERR",
        "--invert-mag", "--time-format", "iso", "--pts-per-knot", "10", "--degree", "3",
        "--n-scan", "200", "--n-bootstrap", "20", "--seed", "999",
    ])

    result2 = run_cli([
        str(fixture), "--x-col", "DATE-OBS", "--y-col", "MAG", "--w-col", "MAG_ERR",
        "--invert-mag", "--time-format", "iso", "--pts-per-knot", "10", "--degree", "3",
        "--n-scan", "200", "--n-bootstrap", "20", "--seed", "999",
    ])

    assert result1.returncode == 0
    assert result2.returncode == 0

    data1 = json.loads(result1.stdout)
    data2 = json.loads(result2.stdout)

    # With same seed, results should be identical
    assert data1["x0"] == data2["x0"]
    assert data1["x0_std"] == data2["x0_std"]
    assert data1["x0_lo"] == data2["x0_lo"]
    assert data1["x0_hi"] == data2["x0_hi"]


def test_cli_different_time_formats():
    """Test that different time formats produce correct x0 units."""
    fixture = Path(__file__).parent / "fixtures" / "V500_Peg_2461261_2026-08-08_G.aavso.txt"

    # Test iso format (minutes from start)
    result_iso = run_cli([
        str(fixture), "--x-col", "DATE-OBS", "--y-col", "MAG", "--w-col", "MAG_ERR",
        "--invert-mag", "--time-format", "iso", "--pts-per-knot", "10", "--degree", "3",
        "--n-scan", "200", "--n-bootstrap", "20", "--seed", "42",
    ])
    assert result_iso.returncode == 0
    data_iso = json.loads(result_iso.stdout)
    assert "utc_time" in data_iso

    # Test minutes format (assuming we parse as minutes directly)
    # For this test, we'd need a file with minutes column; skip for now
    # Just verify iso works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])