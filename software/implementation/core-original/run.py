"""Sequential recoverable real stages; evaluation labels only after whole freezes."""

import argparse, subprocess, sys, time, traceback
from pathlib import Path
from common import *


def stage(root, name, command):
    out = root / "orchestration"
    n, r = attempt(
        out / name,
        {"command": command, "release": digest(root / "release-identity.json")},
    )
    save_json(
        root / "progress.json", {"stage": name, "status": "running", "attempt": n}
    )
    print("REAL STAGE", name, flush=True)
    with (out / name / f"attempt-{n}.log").open("x") as f:
        p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    r.update(
        exit_code=p.returncode,
        status="completed" if p.returncode == 0 else "failed",
        finished=time.time(),
    )
    save_json(out / name / f"attempt-{n}.json", r)
    if p.returncode:
        raise RuntimeError("Preserved stage failure: " + name)


def main(root):
    root = Path(root)
    identity = read(root / "release-identity.json")
    cfg = read(root / "config.json")
    python = sys.executable
    verify_files(root / "code", identity["code"])
    check(root / "config.json", identity["config_sha256"])
    check(root / "method.md", identity["method_sha256"])
    check(root / "environment.lock", identity["environment_sha256"])
    # Physical sources once at launch, again at the post-freeze target boundary.
    import legacy_prepare as legacy

    legacy.verify_registry(root)
    blocks = []
    for fold in cfg["folds"]:
        block = root / "blocks" / fold["name"]
        block.mkdir(parents=True, exist_ok=True)
        binding = {
            "release_sha256": digest(root / "release-identity.json"),
            "code": identity["code"],
            "config_sha256": identity["config_sha256"],
        }
        save_json(block / "release-binding.json", binding)
        if not (block / "config.json").exists():
            save_json(block / "config.json", cfg)
        check(block / "config.json", identity["config_sha256"])
        if not (block / "inputs/prepared.json").exists():
            stage(
                root,
                fold["name"] + "-prepare",
                [
                    python,
                    "-u",
                    str(root / "code/snapshot.py"),
                    "--block",
                    str(block),
                    "--parent",
                    str(Path(cfg["parent_root"]) / "blocks" / fold["name"]),
                ],
            )
        if not (block / "core-frozen.json").exists():
            stage(
                root,
                fold["name"] + "-core",
                [
                    python,
                    "-u",
                    str(root / "code/fit.py"),
                    "--block",
                    str(block),
                    "--stage",
                    "core",
                ],
            )
        for task in TASKS:
            if not (block / "selector" / task / "frozen.json").exists():
                stage(
                    root,
                    fold["name"] + "-select-" + task,
                    [
                        python,
                        "-u",
                        str(root / "code/selector.py"),
                        "--block",
                        str(block),
                        "--task",
                        task,
                    ],
                )
        blocks.append(
            {
                "block": fold["name"],
                "core_sha256": digest(block / "core-frozen.json"),
                "selectors": {
                    t: digest(block / "selector" / t / "frozen.json") for t in TASKS
                },
            }
        )
        save_json(root / "core-progress.json", blocks)
    save_json(
        root / "core-all-predictions-frozen.json",
        {
            "release_sha256": digest(root / "release-identity.json"),
            "blocks": blocks,
            "evaluation_labels_read": False,
        },
    )
    stage(
        root,
        "core-score",
        [
            python,
            "-u",
            str(root / "code/score.py"),
            "--root",
            str(root),
            "--stage",
            "core",
        ],
    )
    for research in ("players", "removals"):
        records = []
        for fold in cfg["folds"]:
            block = root / "blocks" / fold["name"]
            if not (block / f"{research}-frozen.json").exists():
                stage(
                    root,
                    fold["name"] + "-" + research,
                    [
                        python,
                        "-u",
                        str(root / "code/fit.py"),
                        "--block",
                        str(block),
                        "--stage",
                        research,
                    ],
                )
            records.append(
                {
                    "block": fold["name"],
                    "sha256": digest(block / f"{research}-frozen.json"),
                }
            )
        save_json(
            root / f"{research}-all-predictions-frozen.json",
            {
                "release_sha256": digest(root / "release-identity.json"),
                "blocks": records,
                "evaluation_labels_read": False,
                "evaluation_labels_already_accessed_for_core": True,
                "secondary_method_frozen_before_core_scoring": True,
            },
        )
        stage(
            root,
            research + "-score",
            [
                python,
                "-u",
                str(root / "code/score.py"),
                "--root",
                str(root),
                "--stage",
                research,
            ],
        )
    stage(
        root,
        "report",
        [python, "-u", str(root / "code/report.py"), "--root", str(root)],
    )
    save_json(
        root / "completion.json",
        {
            "status": "all_planned_stages_complete",
            "finished": time.time(),
            "core": digest(root / "report/core/results.json"),
            "players": digest(root / "report/players/results.json"),
            "removals": digest(root / "report/removals/results.json"),
            "tests_run": False,
            "synthetic_rehearsal_run": False,
        },
    )
    save_json(root / "progress.json", {"status": "complete"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    main(a.root)
