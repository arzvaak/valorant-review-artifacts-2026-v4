"""Render prior and revision figures plus aggregate CSV views from bundled inputs."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent
RENDERED = ROOT / "rendered"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    run([sys.executable, "render.py"])
    run([
        sys.executable, "code/review_revision_figures.py",
        "--diagnostics", "data/revision/diagnostics",
        "--feedback-results", "data/revision/feedback-results/results.json",
        "--online-calibration", "data/revision/online-calibration/results.json",
        "--output", "rendered/revision-20261016",
    ])
    run([
        sys.executable, "code/review_revision_secondary_table.py",
        "--results", "data/revision/secondary-metrics/results.json",
        "--output", "rendered/revision-20261016/secondary-metrics-tables.md",
    ])
    run([sys.executable, "export_aggregate_tables.py"])
    outputs = sorted(path for path in RENDERED.rglob("*") if path.is_file() and path.name != "reconstruction-receipt.json")
    input_files = sorted(path for path in (ROOT / "data").rglob("*.json") if path.name != "projection-receipt.json")
    receipt = {
        "qualification": "single local aggregate-only reconstruction",
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "inputs": {path.relative_to(ROOT).as_posix(): sha(path) for path in input_files},
        "code": {
            "render_all.py": sha(Path(__file__)),
            "review_revision_figures.py": sha(ROOT / "code/review_revision_figures.py"),
            "review_revision_diagnostics.py": sha(ROOT / "code/review_revision_diagnostics.py"),
            "review_revision_secondary_table.py": sha(ROOT / "code/review_revision_secondary_table.py"),
            "export_aggregate_tables.py": sha(ROOT / "export_aggregate_tables.py"),
            "core_render.py": sha(ROOT / "render.py"),
        },
        "outputs": {path.relative_to(RENDERED).as_posix(): sha(path) for path in outputs},
        "figure_stems": sorted({path.stem for path in outputs if path.suffix == ".png"}),
        "aggregate_table_csv_count": sum(path.suffix == ".csv" for path in outputs),
        "scope": "saved aggregate tables, secondary-metric table and figures only; no new fitting, scoring, resampling, or outcome access",
    }
    (RENDERED / "reconstruction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Aggregate reconstruction complete: {len(receipt['figure_stems'])} figure stems, {receipt['aggregate_table_csv_count']} aggregate CSVs.")


if __name__ == "__main__":
    main()
