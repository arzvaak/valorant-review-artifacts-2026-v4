"""Separate target projection, gated by complete-cohort saved predictions.

Never invoked by the feature or inference runtimes. Score tokens are local
temporaries; artifacts contain binary labels or reason-only quarantine.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from selectolax.parser import HTMLParser
from later_reference_state import read, sha


def direct_targets(body, context):
    html = HTMLParser(body)
    status = [n.text(strip=True).casefold() for n in html.css(".match-header-vs-note")]
    status = [s for s in status if s and not re.fullmatch(r"bo\d+", s, re.I)]
    if status != ["final"]:
        raise ValueError("source_status_not_unambiguously_final")
    panels = {}
    for node in html.css("div.vm-stats-game"):
        gid = node.attributes.get("data-game-id", "")
        if gid == "all":
            continue
        if not gid.isdigit() or gid in panels:
            raise ValueError("source_map_inventory_invalid")
        panels[gid] = node
    if set(panels) != set(context["archived_game_ids"]):
        raise ValueError("source_map_inventory_changed_after_context_freeze")
    labels, reasons = {}, {}
    for gid, panel in panels.items():
        tokens = [
            n.text(strip=True) for n in panel.css(".vm-stats-game-header .team .score")
        ]
        if len(tokens) != 2 or any(re.fullmatch(r"\d+", s) is None for s in tokens):
            reasons[gid] = "invalid_or_missing_map_score"
            continue
        first, second = map(int, tokens)
        if first == second:
            reasons[gid] = "tied_map_score"
            continue
        source_a_won = first > second
        labels[gid] = float(
            not source_a_won if context["orientation_reversed"] else source_a_won
        )
    return labels, reasons


def archive_targets(body, context):
    payload = json.loads(body)["data"]
    if str(payload.get("status", "")).casefold() != "final":
        raise ValueError("source_status_not_unambiguously_final")
    panels = {}
    for game in payload["maps"]:
        gid = str(game["game_id"])
        if not gid.isdigit() or gid in panels:
            raise ValueError("source_map_inventory_invalid")
        panels[gid] = game
    if set(panels) != set(context["archived_game_ids"]):
        raise ValueError("source_map_inventory_changed_after_context_freeze")
    labels, reasons = {}, {}
    for gid, panel in panels.items():
        scores = panel.get("score", {})
        first, second = scores.get("team1"), scores.get("team2")
        if type(first) is not int or type(second) is not int or min(first, second) < 0:
            reasons[gid] = "invalid_or_missing_map_score"
            continue
        if first == second:
            reasons[gid] = "tied_map_score"
            continue
        source_a_won = first > second
        labels[gid] = float(
            not source_a_won if context["orientation_reversed"] else source_a_won
        )
    return labels, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--forecasts", type=Path, required=True)
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "Preserve all target attempts"
    context, features, forecasts = (
        read(args.context / "projection.json"),
        read(args.features / "frozen.json"),
        read(args.forecasts / "frozen.json"),
    )
    assert context["status"] == "complete_real_source_context_projection"
    assert features["complete_cohort"] and forecasts["complete_cohort"]
    assert forecasts["status"] == "predictions_frozen_before_new_targets"
    assert not forecasts["new_targets_read"]
    assert (
        context["method_sha256"]
        == features["method_sha256"]
        == forecasts["method_sha256"]
        == sha(args.method)
    )
    assert features["context_projection_sha256"] == sha(
        args.context / "projection.json"
    )
    assert forecasts["features_sha256"] == sha(args.features / "frozen.json")
    for root, receipt in (
        (args.context, context),
        (args.features, features),
        (args.forecasts, forecasts),
    ):
        for name, h in receipt["files"].items():
            assert sha(root / name) == h
    method = read(args.method)
    series = {r["match_id"]: r for r in read(args.context / "series-context.json")}
    maps = {r["game_id"]: r for r in read(args.context / "map-context.json")}
    expected = {("series_winner", r["game_id"]) for r in series.values()} | {
        ("map_winner", r["game_id"]) for r in maps.values()
    }
    keys = read(args.forecasts / "keys.json")
    assert len(keys) == len(expected) and set(map(tuple, keys)) == expected
    sources = {r["receipt_sha256"]: r for r in read(args.context / "sources.json")}
    # Complete physical identity checks precede every new target token read.
    for r in sources.values():
        assert sha(r["receipt_path"]) == r["receipt_sha256"]
        assert sha(r["raw_path"]) == r["raw_sha256"]
    by_uid, quarantine = {}, []
    for mid, row in series.items():
        assert (
            row["event_id"] in method["event_allowlist"]
            and row["event_id"] not in method["protected_event_denylist"]
        )
        rec = sources[row["source_receipt_sha256"]]
        source_receipt = read(rec["receipt_path"])
        assert (
            source_receipt["match_id"] == mid
            and source_receipt["event_id"] == row["event_id"]
        )
        assert row["raw_sha256"] == rec["raw_sha256"]
        try:
            labels, reasons = (
                archive_targets
                if source_receipt["route"] == "archive"
                else direct_targets
            )(Path(rec["raw_path"]).read_bytes(), row)
        except (ValueError, KeyError, TypeError):
            labels, reasons = {}, {
                g: "invalid_final_status_or_exact_inventory"
                for g in row["archived_game_ids"]
            }
        for gid in row["archived_game_ids"]:
            if gid in maps:
                assert maps[gid]["match_id"] == mid
                if gid in labels:
                    by_uid["map_winner", gid] = labels[gid]
                else:
                    quarantine.append(
                        {
                            "task": "map_winner",
                            "game_id": gid,
                            "event_id": row["event_id"],
                            "reason": reasons.get(gid, "map_label_unresolved"),
                        }
                    )
        games = row["archived_game_ids"]
        if not games or any(g not in labels for g in games):
            quarantine.append(
                {
                    "task": "series_winner",
                    "game_id": row["game_id"],
                    "event_id": row["event_id"],
                    "reason": "incomplete_exact_map_target_set",
                }
            )
        else:
            won = sum(labels[g] for g in games)
            if won * 2 == len(games):
                quarantine.append(
                    {
                        "task": "series_winner",
                        "game_id": row["game_id"],
                        "event_id": row["event_id"],
                        "reason": "series_tied_map_majority",
                    }
                )
            else:
                by_uid["series_winner", row["game_id"]] = float(won * 2 > len(games))
    target = np.array([by_uid.get(tuple(k), np.nan) for k in keys], dtype="float32")
    args.output.mkdir(parents=True)
    np.save(args.output / "labels.npy", target)
    for name, rows in [("keys.json", keys), ("quarantine.json", quarantine)]:
        (args.output / name).write_text(json.dumps(rows, indent=2) + "\n")
    receipt = {
        "status": "targets_projected_after_complete_forecast_freeze",
        "method_sha256": sha(args.method),
        "context_projection_sha256": sha(args.context / "projection.json"),
        "forecast_freeze_sha256": sha(args.forecasts / "frozen.json"),
        "code_sha256": sha(__file__),
        "valid_targets": int(np.isfinite(target).sum()),
        "quarantined_targets": int((~np.isfinite(target)).sum()),
        "raw_scores_saved": False,
        "selection_or_fit": False,
        "target_definition": "individual map score winner; series strict majority over complete frozen observed map inventory",
        "files": {p.name: sha(p) for p in args.output.iterdir()},
    }
    (args.output / "frozen.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "valid_targets": receipt["valid_targets"],
                "quarantined_targets": receipt["quarantined_targets"],
            }
        )
    )


if __name__ == "__main__":
    main()
