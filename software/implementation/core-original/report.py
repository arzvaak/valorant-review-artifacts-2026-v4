"""Editable scientific figures, complete tables and separate expanded manuscript."""

import argparse, csv, shutil
from pathlib import Path
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def table(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(
            {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def main(root):
    root = Path(root)
    report = root / "report"
    figs = report / "figures"
    figs.mkdir(exist_ok=True)
    core = read(report / "core/results.json")
    others = [read(report / s / "results.json") for s in ("players", "removals")]
    for result in [core, *others]:
        out = report / result["stage"]
        table(out / "metrics.csv", result["metrics"])
        table(out / "contrasts.csv", result["contrasts"])
        table(out / "seed-metrics.csv", result["seed_metrics"])
        table(out / "failure-ledger.csv", result["failures_and_fallbacks"])
        table(out / "denominators.csv", result["denominators"])
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    for task in TASKS:
        rows = [
            r
            for r in core["metrics"]
            if r["task"] == task
            and r["scope"] == "broad_nonglobal"
            and r["mode"] == "calibrated"
        ]
        rows = sorted(rows, key=lambda r: (r["brier"], r["package"]))
        fig, ax = plt.subplots(figsize=(9, 12))
        ax.scatter([r["brier"] for r in rows], range(len(rows)))
        ax.set_yticks(range(len(rows)), [r["package"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Event-equal Brier score (lower is better)")
        ax.set_xlim(0, max(0.3, max(r["brier"] for r in rows) * 1.1))
        ax.set_title(task.replace("_", " ") + " • calibrated development evaluation")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        for ext in ("svg", "pdf"):
            fig.savefig(figs / (task + "-benchmark." + ext))
        plt.close(fig)
        rows = [
            r
            for r in core["contrasts"]
            if r["task"] == task
            and r["metric"] == "brier"
            and r["correction_family"] == "primary120"
        ]
        fig, ax = plt.subplots(figsize=(10, 10))
        rows = sorted(rows, key=lambda r: r["id"])
        ax.axvline(0, color="gray", lw=1)
        for i, r in enumerate(rows):
            ax.plot(r["ci95"], [i, i], color="black", lw=1)
            ax.scatter(
                r["difference"],
                i,
                color="#176b47" if r["conservative_rejection"] else "#777777",
                s=20,
            )
        ax.set_yticks(
            range(len(rows)),
            [r["id"].split("/")[1] + "/" + r["id"].split("/")[2] for r in rows],
        )
        ax.invert_yaxis()
        ax.set_xlabel(
            "Paired event-equal Brier difference; negative favors added information"
        )
        ax.set_title(
            "Pointwise 95% intervals; green passes conservative 120-family Holm rule"
        )
        fig.tight_layout()
        for ext in ("svg", "pdf"):
            fig.savefig(figs / (task + "-primary-contrasts." + ext))
        plt.close(fig)
    save_json(
        figs / "specification.json",
        {
            "inputs": {
                r["stage"]: digest(report / r["stage"] / "results.json")
                for r in [core, *others]
            },
            "filter": "broad_nonglobal calibrated; benchmark all packages, contrast primary120 Brier",
            "mapping": "x Brier or paired difference; y model/contrast identity",
            "scales": "benchmark begins zero; contrast linear with zero reference",
            "uncertainty": "pointwise complete-event bootstrap 95%; conservative Holm color",
            "outputs": files(figs),
            "code_sha256": digest(Path(__file__)),
        },
    )
    lines = [
        "# Expanded VALORANT predictor study",
        "",
        "This is a retrospective development study of two distinct horizons: map winners after both compositions are known, and series winners before play using an explicitly inferred earlier-roster proxy. After exploratory FT selection diagnostics and a compute assessment, it compares five established core model families. FT-Transformer is deferred; its completed models and checkpoints remain preserved outside active selection and scoring. It separates general numeric history, composition/tendencies and agent-linked history.",
        "",
        "The primary population is the broad nonglobal archive, including explicit unknown classifications. It is not verified professional-only evidence. Reviewed regional circuits and known global tournaments are secondary scopes. Predictions precede evaluation-label projection in this release; source timing uses a schedule-plus-120-minute availability proxy and does not establish prospective validation. E12 and locked confirmation are excluded.",
        "",
        "## Method",
        "",
        "Whole events and Match IDs are separated chronologically. Historical state uses only permitted training rows, freezes equal-time batches, and requires history_end_timestamp strictly before prediction cutoff. Training-only imputation, vocabularies, missingness and mirrored side symmetry apply to all families. Four nonlinear configurations and three selected seeds compete; logistic has nine penalties. Stopping, model selection and scalar calibration use distinct events. The operational selector compares the top three validation packages with K20 and finite pairwise blends.",
        "",
        "The five-family core has 100 primary and 40 linked contrasts. Historical correction-family identifiers and conservative Monte Carlo bounds retain their original 120/48 budgets; Holm adjusts the actual retained comparisons. Primary calibrated information contrasts share one Holm family. Linked history (48), operational prediction (12), individual player sets (48) and conditional removals (40) are separate declared families. CUDA float64 complete-event bootstrap uses 131,072 draws with conservative simultaneous Monte Carlo upper bounds. Intervals condition on fitted forecasts and do not include training-refit uncertainty. Year omissions and moving event blocks are descriptive.",
        "",
        "## Results",
        "",
    ]
    for task in TASKS:
        rows = sorted(
            [
                r
                for r in core["metrics"]
                if r["task"] == task
                and r["scope"] == "broad_nonglobal"
                and r["mode"] == "calibrated"
            ],
            key=lambda r: (r["brier"], r["package"]),
        )
        selector = next(r for r in rows if r["package"] == "selector")
        elo = next(r for r in rows if r["package"] == "elo_k20")
        best = rows[0]
        lines += [
            f"{task}: the frozen operational selector scored Brier {selector['brier']:.6f}, log loss {selector['log_loss']:.6f}, and accuracy {selector['accuracy']:.2%} over {selector['rows']:,} rows in {selector['events']:,} complete events. Calibrated K20 scored {elo['brier']:.6f} and {elo['log_loss']:.6f}. The lowest outer Brier package was {best['package']} ({best['brier']:.6f}); this retrospective ranking did not select the operational model.",
            "",
        ]
    for result in [core, *others]:
        for family in sorted({r["correction_family"] for r in result["contrasts"]}):
            rows = [r for r in result["contrasts"] if r["correction_family"] == family]
            supported = [r for r in rows if r.get("conservative_rejection")]
            lines += [
                f"{family}: {len(supported)} of {len(rows)} declared contrasts passed the conservative Holm rule. Complete estimates, directions and pointwise intervals are in {result['stage']}/contrasts.csv; a rejection can indicate benefit or harm.",
                "",
            ]
        fallback = sum(not r["trained"] for r in result["failures_and_fallbacks"])
        failed = sum(
            len(r["failed_candidates"]) for r in result["failures_and_fallbacks"]
        )
        lines += [
            f"{result['stage']} retained {fallback} fallback packages and {failed} failed candidates. Their reasons and stage denominators are included in the failure ledger.",
            "",
        ]
    lines += [
        "## Interpretation and limitations",
        "",
        "General form, conditional agent proficiency and composition usage are distinct inputs. Team planning is unmeasured, so this study cannot identify its effect. Conditional removals assess model reliance rather than causal mechanisms. Set-versus-summary contrasts change both representation and architecture. Unknown tiers, historical roster proxies, missing source statistics, timing proxies, GPU nondeterminism and temporal drift limit deployment claims. Nonsignificance does not establish equivalence. Unfavorable and neutral results remain in the complete tables.",
        "",
        "## Availability and reproducibility",
        "",
        f"Release identity: {root.name}. Source databases, datasets, binaries and run artifacts remain outside Git. The method, code/configuration/environment identities, per-candidate failures, checkpoints, frozen predictions, target quarantine, tables and editable SVG/PDF figures are retained. The completed earlier paper remains unchanged. ",
        "",
        "## Inference bundles",
        "",
        "Each task has a release-local bundle referencing each rolling block’s frozen selection, component models, processors, calibrators, source and input contract. These reproduce saved forecast packages; they do not establish a live source feed or prospective delivery.",
    ]
    text = "\n".join(lines) + "\n"
    (report / "manuscript.md").write_text(text)
    repo = (
        Path(read(root / "config.json")["repository"])
        if "repository" in read(root / "config.json")
        else Path(os.environ.get("VALORANT_REPOSITORY", "."))
    )
    dest = repo / "docs/manuscript/expanded-predictor-study-draft.md"
    dest.write_text(text)
    summary = repo / "docs/reports/strong-predictor-study-results-2026-10-09.md"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(text)
    for task in TASKS:
        bundles = []
        for fold in read(root / "config.json")["folds"]:
            block = root / "blocks" / fold["name"]
            selector = read(block / "selector" / task / "selection.json")
            components = []
            for name in selector["chosen"]["components"]:
                if name == "elo_k20":
                    components.append(
                        {"name": name, "method": "frozen strictly-prior K20 rating"}
                    )
                    continue
                package = block / "packages/core" / task / name
                r = read(package / "frozen.json")
                selection = read(package / "selection.json")
                params = selection["selected"]["record"]["identity"]["parameters"]
                selected = [
                    p
                    for p in r["model_records"]
                    if p["record"]["status"] == "fitted"
                    and p["record"]["identity"]["parameters"] == params
                ]
                components.append(
                    {
                        "name": name,
                        "freeze_sha256": digest(package / "frozen.json"),
                        "processor": str(package / "processor.json"),
                        "models": [
                            {
                                "model": p["record"]["model"],
                                "sha256": p["record"]["model_sha256"],
                                "seed": p["record"]["identity"]["seed"],
                            }
                            for p in selected
                        ],
                    }
                )
            bundles.append(
                {
                    "block": fold["name"],
                    "selector_directory": str(block / "selector" / task),
                    "selector_freeze": digest(
                        block / "selector" / task / "frozen.json"
                    ),
                    "recipe": selector,
                    "components": components,
                    "input_contract": read(block / "inputs/prepared.json")["tasks"][
                        task
                    ],
                }
            )
        save_json(
            root / "inference" / task / "bundle.json",
            {
                "task": task,
                "release_sha256": digest(root / "release-identity.json"),
                "environment_sha256": digest(root / "environment.lock"),
                "blocks": bundles,
                "current_roster_verified": False if task == "series_winner" else None,
                "live_feed_verified": False,
            },
        )
    save_json(
        report / "bundle.json",
        {
            "reports": files(report),
            "inference": files(root / "inference"),
            "manuscript_sha256": digest(dest),
            "release_sha256": digest(root / "release-identity.json"),
        },
    )
    print("REPORT COMPLETE", report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    main(a.root)
