"""Persistent, identity-bound later-evaluation stages after source completion."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from later_reference_state import read, sha


def save(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def wait_source(requirement, method_sha, root):
    while True:
        state = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                requirement["unit"],
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "ExecMainStatus",
            ],
            capture_output=True,
            text=True,
        )
        values = dict(
            line.split("=", 1) for line in state.stdout.splitlines() if "=" in line
        )
        if values.get("ActiveState") in ("active", "activating"):
            save(
                root / "progress.json",
                {
                    "stage": "waiting_for_source_completion",
                    "unit": requirement["unit"],
                    "source_state": values,
                    "new_targets_projected": False,
                    "scored": False,
                },
            )
            time.sleep(15)
            continue
        assert (
            values.get("ExecMainStatus") == "0"
        ), "Source service stopped without verified success"
        receipt = read(requirement["completion"])
        assert receipt["status"] == "source_capture_complete"
        assert receipt["method_sha256"] == method_sha
        assert (
            sha(requirement["job"])
            == requirement["job_sha256"]
            == receipt["job_sha256"]
        )
        assert receipt["source_plan_sha256"] == requirement["source_plan_sha256"]
        return sha(requirement["completion"])


def stage(root, job, name, script, arguments, destination, marker):
    path = root / "stage-receipts" / (name + ".json")
    command = [job["python"], str(root / "code" / script), *map(str, arguments)]
    identity = {
        "command": command,
        "code_sha256": job["code"][script],
        "job_sha256": sha(root / "job.json"),
    }
    if path.exists():
        receipt = read(path)
        assert receipt["identity"] == identity
        assert sha(destination / marker) == receipt["output_receipt_sha256"]
        output = read(destination / marker)
        for f, h in output.get("files", {}).items():
            assert sha(destination / f) == h
        return
    assert (
        not destination.exists()
    ), "Preserve unfinished stage; record a versioned repair instead of overwriting"
    save(
        root / "progress.json",
        {"stage": name, "destination": str(destination), "scored": False},
    )
    save(
        root / "stage-receipts" / (name + "-started.json"),
        {
            "identity": identity,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    with (root / "logs" / (name + ".log")).open("a") as log:
        subprocess.run(
            command,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    output = read(destination / marker)
    for f, h in output.get("files", {}).items():
        assert sha(destination / f) == h
    save(
        path,
        {
            "identity": identity,
            "output": str(destination),
            "output_receipt_sha256": sha(destination / marker),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def report(root, job):
    result = read(root / "scoring/results.json")
    completion = read(root / "scoring/completion.json")
    assert sha(root / "scoring/results.json") == completion["results_sha256"]
    context = read(root / "context/projection.json")
    labels = read(root / "targets/frozen.json")
    lines = [
        "# Later-competition evaluation of retained VALORANT models",
        "",
        "This is a retrospective later-competition comparison of saved historical model packages. It is not prospective prediction, an independently externally executed replication, a latest-cutoff refit, or unqualified professional-population evidence.",
        "",
        f"The complete source plan contains {context['captured_matches']} match records. Eligibility produced {context['map_context_rows']} map contexts and {context['series_context_rows']} series contexts; target projection retained {labels['valid_targets']} labels and quarantined {labels['quarantined_targets']}. Source capture, input eligibility and valid-target denominators are different units.",
        "",
        "The existing source endpoint is 4 September 2026, 15:05 UTC. Later events were selected from the preserved completed-event listing before new target extraction. Residual fitting training ends in July 2025, with stopping/selection/calibration later; the historical online state extends through the admitted existing data. All new unreviewed-cohort observations are score-only and do not update history. Source schedule/lineup availability remains a proxy; actual pre-play publication timing is unverified.",
        "",
        "## Predictive results",
        "",
        "Accuracy uses the saved decisions; Brier and log loss use saved historically calibrated probabilities for residuals and raw reference probabilities. No new outcome changed a model, threshold or probability calibrator.",
        "",
        "| Task | Arm | Valid rows | Events | Row accuracy | Equal-event accuracy | Equal-event Brier | Equal-event log loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["metrics"]:
        if row["status"] != "scored":
            lines.append(
                f"| {row['task']} | {row['arm']} | 0 | 0 | unresolved | unresolved | unresolved | unresolved |"
            )
            continue
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['rows']} | {row['events']} | {row['row_accuracy']:.2%} | {row['equal_event_accuracy']:.2%} | {row['equal_event_brier']:.6f} | {row['equal_event_log_loss']:.6f} |"
        )
    lines += [
        "",
        "## All twelve primary loss comparisons",
        "",
        "Negative differences favor the rich residual arm. Intervals are pointwise 95% complete-event bootstrap intervals conditional on the saved fits. The decision uses simultaneous 99% Monte Carlo upper p bounds and Holm across all twelve contrasts. A non-rejection does not establish equivalence; a point-estimate rank does not establish superiority.",
        "",
        "| Task | Parent | Loss | Difference | Pointwise 95% interval | Conservative Holm p | Decision |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in result["primary_contrasts"]:
        interval = (
            "unresolved"
            if row["ci95"] is None
            else f"[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]"
        )
        difference = (
            "unresolved" if row["difference"] is None else f"{row['difference']:.6f}"
        )
        decision = (
            (
                "supported lower loss"
                if row["difference"] < 0
                else "supported higher loss"
            )
            if row["conservative_rejection"]
            else "no corrected support" if row["status"] == "scored" else row["status"]
        )
        lines.append(
            f"| {row['task']} | {row['parent']} | {row['metric']} | {difference} | {interval} | {row['holm_mc_upper_p']:.6g} | {decision} |"
        )
    lines += [
        "",
        "## Eligibility and event denominators",
        "",
        "Every allowlisted event is retained in this table, including zero eligible/valid cells. See `context/quarantine.json` for input exclusions and `targets/quarantine.json` for label exclusions. A missing map lineup can exclude a map forecast while its series context remains available; series targets require a complete observed map inventory.",
        "",
        "| Task | Event ID | Frozen context rows | Valid targets | Invalid targets |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["denominators"]:
        lines.append(
            f"| {row['task']} | {row['event_id']} | {row['frozen_rows']} | {row['valid_targets']} | {row['invalid_targets']} |"
        )
    lines += [
        "",
        "## Interpretation limits and reproducibility",
        "",
        "The cohort comes from a bounded VLR listing and contains unreviewed competition classifications. Its results do not establish performance for the reviewed regional circuits or all professional VALORANT. E12/locked records and China event 3129 remain excluded. Archived empty player arrays were supplemented by serialized direct capture with identity/time/map-inventory checks. Raw responses and failed/partial attempts are preserved outside Git.",
        "",
        "The exact event-level metrics, pointwise accuracy intervals, ROC-AUC, ECE sensitivity and reporting-only calibration diagnostics are in `scoring/results.json`. Calibration point estimates have no slope/intercept intervals and may be unresolved; they never alter the probabilities. Few events and shared team histories limit cluster independence and inferential power. Neither null results nor unsupported/degenerate bootstrap comparisons prove practical equivalence. The retained-model comparison is descriptive beyond its declared contrast family.",
        "",
        "The source policy records owner-reported written permission for private slow acquisition, not independently inspected correspondence or public redistribution clearance. The six-hour release proxy and assumed after-picks lineup horizon are explicit.  The scoring optimizer fits reporting-only reliability diagnostics on CUDA after prediction freeze; no predictive model or calibrator was fitted on this cohort.",
        "",
        f"Job identity: `{sha(root / 'job.json')}`. Method identity: `{job['method_sha256']}`. Score completion identity: `{sha(root / 'scoring/completion.json')}`. Stage receipts, immutable code, environment and logs are in this release directory.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return sha(root / "report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    args = ap.parse_args()
    job = read(args.job)
    root = args.job.parent
    assert sha(job["method"]) == job["method_sha256"]
    for name, h in job["code"].items():
        assert sha(root / "code" / name) == h
    assert sha(root / "environment.json") == job["environment_sha256"]
    for path, h in job["dependencies"].items():
        assert sha(path) == h
    for name in ("logs", "stage-receipts"):
        (root / name).mkdir(exist_ok=True)
    try:
        source_receipts = {
            r["unit"]: wait_source(r, job["method_sha256"], root)
            for r in job["sources"]
        }
        for path, h in job["dependencies"].items():
            assert sha(path) == h
        save(
            root / "source-ready.json",
            {"completions": source_receipts, "method_sha256": job["method_sha256"]},
        )
        method = job["method"]
        context, reference, features, forecasts, targets, scoring = [
            root / n
            for n in (
                "context",
                "reference",
                "features",
                "forecasts",
                "targets",
                "scoring",
            )
        ]
        stage(
            root,
            job,
            "context",
            "project_later_context.py",
            [
                "--capture-root",
                job["capture_root"],
                "--supplemental-root",
                job["supplemental_root"],
                "--require-complete",
                "--method",
                method,
                "--output",
                context,
            ],
            context,
            "projection.json",
        )
        stage(
            root,
            job,
            "reference",
            "later_reference_state.py",
            ["--context", context, "--method", method, "--output", reference],
            reference,
            "frozen.json",
        )
        stage(
            root,
            job,
            "features",
            "later_residual_features.py",
            [
                "--context",
                context,
                "--reference",
                reference,
                "--method",
                method,
                "--output",
                features,
            ],
            features,
            "frozen.json",
        )
        stage(
            root,
            job,
            "forecasts",
            "later_retained_inference.py",
            [
                "--features",
                features,
                "--reference",
                reference,
                "--method",
                method,
                "--output",
                forecasts,
            ],
            forecasts,
            "frozen.json",
        )
        stage(
            root,
            job,
            "targets",
            "later_targets.py",
            [
                "--context",
                context,
                "--features",
                features,
                "--forecasts",
                forecasts,
                "--method",
                method,
                "--output",
                targets,
            ],
            targets,
            "frozen.json",
        )
        stage(
            root,
            job,
            "scoring",
            "later_score.py",
            [
                "--context",
                context,
                "--forecasts",
                forecasts,
                "--targets",
                targets,
                "--method",
                method,
                "--output",
                scoring,
            ],
            scoring,
            "completion.json",
        )
        report_sha = report(root, job)
        save(
            root / "completion.json",
            {
                "status": "all_later_evaluation_stages_complete",
                "job_sha256": sha(args.job),
                "method_sha256": job["method_sha256"],
                "source_ready_sha256": sha(root / "source-ready.json"),
                "stage_receipts": {
                    p.name: sha(p)
                    for p in sorted((root / "stage-receipts").glob("*.json"))
                    if not p.name.endswith("-started.json")
                },
                "report_sha256": report_sha,
                "score_completion_sha256": sha(scoring / "completion.json"),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        save(
            root / "progress.json",
            {"stage": "complete", "scored": True, "report": str(root / "report.md")},
        )
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "job_sha256": sha(args.job),
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "prior_progress": (
                read(root / "progress.json")
                if (root / "progress.json").exists()
                else None
            ),
        }
        save(root / ("failure-" + str(time.time_ns()) + ".json"), failure)
        save(root / "progress.json", {"stage": "failed", "failure": failure})
        raise


if __name__ == "__main__":
    main()
