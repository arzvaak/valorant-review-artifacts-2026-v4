"""Construct retained residual inputs from admitted old source histories only."""

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import os
import sys

import duckdb
import numpy as np
import yaml
from later_reference_state import read, sha


def linked_series(history, row, cutoff, old):
    records = []
    for side, team in enumerate((row["team_a_id"], row["team_b_id"])):
        roster, _ = history.rosters.get(team, (None, None))
        for player in sorted(roster or []):
            counts = {
                a: s.maps for a, s in history.player_agent[player].items() if s.maps
            }
            agent = min(counts, key=lambda a: (-counts[a], a)) if counts else None
            linked = history.player_agent[player][agent] if agent else old.Mean()
            recent = old.Mean()
            for when, mp, a, values in history.recent[player]:
                if cutoff - 90 * 86400 < when < cutoff and a == agent:
                    recent.add(values)
            records.append((side, linked, recent))
    replacements = {}
    for side, suffix in enumerate(("a", "b")):
        own = [(linked, recent) for z, linked, recent in records if z == side]
        for field in ("first_kills", "first_deaths", "opening_balance"):
            replacements["matched_" + field + "_" + suffix] = old.finite_mean(
                [s.value(field) for s, r in own]
            )
        replacements["matched_player_map_agent_maps_" + suffix] = old.finite_mean(
            [s.maps for s, r in own]
        )
        for field in ("ACS", "ADR", "KAST", "opening_balance"):
            replacements["recent90_matched_" + field + "_" + suffix] = old.finite_mean(
                [r.value(field) for s, r in own]
            )
    return replacements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "Preserve prior feature attempts"
    method, context = read(args.method), read(args.context / "projection.json")
    reference = read(args.reference / "frozen.json")
    assert context["method_sha256"] == reference["method_sha256"] == sha(args.method)
    assert reference["context_projection_sha256"] == sha(
        args.context / "projection.json"
    )
    for root, receipt in ((args.context, context), (args.reference, reference)):
        for f, h in receipt["files"].items():
            assert sha(root / f) == h
    core = Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "strong-predictors-20261010-v6"
    dynamic = (
        Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "dynamic-predictor-20261010-v1"
    )
    cfg = read(core / "config.json")
    prepared_path = core / "blocks/rolling_12/inputs/prepared.json"
    prepared = read(prepared_path)
    for package in method["parent_model_packages"]:
        assert package["source_prepared"] == sha(prepared_path)
    parent_prepared_path = Path(prepared["parent_root"]) / "inputs/prepared.json"
    assert sha(parent_prepared_path) == prepared["parent_prepared_sha256"]
    parent_prepared = read(parent_prepared_path)
    old_path = core / "code/legacy_prepare.py"
    spec = importlib.util.spec_from_file_location("later_legacy", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    records = read(dynamic / "online/sources.json")
    assert (
        sha(dynamic / "online/sources.json")
        == read(dynamic / "online/frozen.json")["sources"]
    )
    meta, refs, arrays = {}, {}, {}
    for record in records:
        assert (
            sha(record["keys"]) == record["keys_sha256"]
            and sha(record["labels"]) == record["labels_sha256"]
        )
        arrays[record["labels"]] = np.load(record["labels"], mmap_mode="r")
        if Path(record["keys"]).parent.name != "map_winner":
            continue
        for i, k in enumerate(read(record["keys"])):
            meta.setdefault(k["game_id"], k)
            refs.setdefault(k["game_id"], (record["labels"], i))
    registry = Path(cfg["registry_root"])
    registry_receipt = read(registry / "release.json")
    for f in ("maps.json", "target-plan.json"):
        assert sha(registry / f) == registry_receipt["files"][f]
    registry_rows = {r["game_id"]: r for r in read(registry / "maps.json")}
    target_plan = read(registry / "target-plan.json")
    labels = {}
    for g, k in meta.items():
        if k["region"] == "global":
            continue
        p, i = refs[g]
        y = float(arrays[p][i])
        if math.isfinite(y):
            assert y in (0.0, 1.0) and registry_rows[g]["anchor"] == k["anchor"]
            labels[g] = y
    rows = [registry_rows[g] for g in sorted(labels)]
    ids = [r["game_id"] for r in rows]
    for p, h in (
        (cfg["source_database_path"], cfg["source_database_sha256"]),
        (cfg["database_path"], cfg["database_sha256"]),
    ):
        assert sha(p) == h
    source = duckdb.connect(cfg["source_database_path"], read_only=True)
    warehouse = duckdb.connect(cfg["database_path"], read_only=True)
    source.execute("set threads=2")
    warehouse.execute("set threads=2")
    lineups = collections.defaultdict(list)
    for g, t, p, a in source.execute(
        "select game_id,team_id,player_id,agent_alias from core.lineups where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        lineups[g].append((t, p, a))
    rounds = collections.defaultdict(list)
    for g, n, t, h in source.execute(
        "select game_id,round_number,winner_team_id,raw_sha256 from main.map_rounds where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        rounds[g].append((n, t, h))
    valid, reasons = {}, {}
    for row in rows:
        g = row["game_id"]
        seq = rounds[g]
        bound = {z["parent_sha256"] for z in target_plan[g]}
        n = len(seq)
        wins = sum(t == row["team_a_id"] for _, t, _ in seq)
        if (
            not n
            or sorted(z[0] for z in seq) != list(range(1, n + 1))
            or any(
                t not in (row["team_a_id"], row["team_b_id"]) or h not in bound
                for _, t, h in seq
            )
            or wins * 2 == n
            or int(wins * 2 > n) != labels[g]
        ):
            reasons[g] = "missing_or_invalid_source_bound_round_sequence"
        else:
            valid[g] = (float(2 * wins - n), wins / n, n)
    del rounds
    print("LOAD_ADMITTED_OLD_PERFORMANCE", len(rows), flush=True)
    observed, teamobs, duels, conflicts = old.load_observations(
        warehouse, source, rows, lineups, valid
    )
    source.close()
    warehouse.close()
    print("OLD_PERFORMANCE_LOADED", flush=True)
    trainkeys_path = core / "blocks/rolling_12/inputs/map_winner/train-keys.json"
    trainlabels_path = core / "blocks/rolling_12/inputs/map_winner/train-labels.npy"
    assert sha(trainkeys_path) == prepared["files"]["map_winner/train-keys.json"]
    assert sha(trainlabels_path) == prepared["files"]["map_winner/train-labels.npy"]
    trainkeys, trainlabels = read(trainkeys_path), np.load(trainlabels_path)
    training = {k["game_id"]: float(y) for k, y in zip(trainkeys, trainlabels)}
    assert all(g in labels and labels[g] == y for g, y in training.items())
    trainrows = sorted(
        [registry_rows[g] for g in training], key=lambda r: (r["anchor"], r["game_id"])
    )
    base = old.History(
        parent_prepared["agent_vocabulary_training_only"],
        parent_prepared["map_vocabulary_training_only"],
    )
    base.player_agent = collections.defaultdict(
        lambda: collections.defaultdict(old.Mean)
    )
    online = old.History(
        sorted({a.lower() for slots in lineups.values() for t, p, a in slots}),
        sorted({r["map_entity_id"] for r in registry_rows.values()}),
    )
    pending = sorted(rows, key=lambda r: r["anchor"])
    alias_path = Path("configs/entities/vlr-agents-9e9ec894.yaml").resolve()
    alias_by_entity = collections.defaultdict(set)
    observed_aliases = {a.lower() for slots in lineups.values() for t, p, a in slots}
    for r in yaml.safe_load(alias_path.read_text())["agents"]:
        if (
            r["review_state"] == "reviewed"
            and r["observed_alias"].lower() in observed_aliases
        ):
            alias_by_entity[r["agent_entity_id"]].add(r["observed_alias"].lower())
    contexts = {}
    for name in ("map-context.json", "series-context.json"):
        for k in read(args.context / name):
            assert (
                not k["feedback_allowed"] and k["event_id"] in method["event_allowlist"]
            )
            assert k["match_id"] not in {r["match_id"] for r in rows}
            contexts[k["task"], k["game_id"]] = k
            if k["task"] == "map_winner":
                assert k["game_id"] not in lineups
                for slot in k["lineups"]:
                    aliases = alias_by_entity[slot["agent_entity_id"]]
                    assert len(aliases) == 1
                    lineups[k["game_id"]].append(
                        (slot["team_id"], slot["player_id"], next(iter(aliases)))
                    )
    reference_keys = [tuple(k) for k in read(args.reference / "keys.json")]
    with np.load(args.reference / "features.npz") as data:
        dynamic_features = {
            key: row.copy() for key, row in zip(reference_keys, data["dynamic"])
        }
    assert set(dynamic_features) == set(contexts)
    original_inputs, rich_inputs, audits = {}, {}, []
    base_cursor, cursor = 0, 0
    for uid, k in sorted(
        contexts.items(),
        key=lambda item: (item[1]["prediction_cutoff"], item[0][1], item[0][0]),
    ):
        cutoff = k["prediction_cutoff"]
        while (
            base_cursor < len(trainrows)
            and trainrows[base_cursor]["anchor"] + 7200 < cutoff
        ):
            end = base_cursor + 1
            while (
                end < len(trainrows)
                and trainrows[end]["anchor"] == trainrows[base_cursor]["anchor"]
            ):
                end += 1
            batch = trainrows[base_cursor:end]
            base.update(batch, lineups, observed, teamobs, duels, training, 7200)
            for row in batch:
                for t, p, a in lineups[row["game_id"]]:
                    base.player_agent[p][a.lower()].add(
                        observed.get((row["game_id"], t, p), {})
                    )
            base_cursor = end
        while cursor < len(pending) and pending[cursor]["anchor"] + 21600 < cutoff:
            end = cursor + 1
            while (
                end < len(pending)
                and pending[end]["anchor"] == pending[cursor]["anchor"]
            ):
                end += 1
            online.update(
                pending[cursor:end], lineups, observed, teamobs, duels, labels, 21600
            )
            cursor = end
        assert base.last is None or base.last < cutoff
        assert online.last is None or online.last < cutoff
        series = uid[0] == "series_winner"
        values, composition, columns = base.features(k, lineups, cutoff, series)
        refreshed, _, rich_columns = online.features(k, lineups, cutoff, series)
        definition = prepared["tasks"][uid[0]]
        assert columns.names == rich_columns.names == definition["old_names"]
        assert len(composition) == definition["composition_width"]
        cells = dict(zip(columns.names, values))
        if series:
            cells.update(linked_series(base, k, cutoff, old))
        names = definition["feature_names"]
        numeric = np.array([cells.get(name, np.nan) for name in names], dtype="float32")
        for j, name in enumerate(names):
            if name in cells:
                continue
            assert "_coverage_" in name
            family, group, side = (
                definition["semantic"][j],
                definition["groups"][j],
                name[-1],
            )
            cc = [
                i
                for i, nm in enumerate(names)
                if definition["semantic"][i] == family
                and definition["groups"][i] == group
                and nm.endswith("_" + side)
                and "_coverage_" not in nm
            ]
            numeric[j] = np.isfinite(numeric[cc]).mean() if cc else 0
        numeric[names.index("elo_log_odds_difference")] = dynamic_features[uid][0]
        rich_numeric = numeric.copy()
        rich_cells = dict(zip(rich_columns.names, refreshed))
        for j, name in enumerate(names):
            if (
                definition["semantic"][j] in ("stars", "economy", "pistols")
                or name.startswith("recent90_ACS")
            ) and name in rich_cells:
                rich_numeric[j] = rich_cells[name]
        for j, name in enumerate(names):
            if "_coverage_" not in name or name.split("_")[0] not in (
                "stars",
                "economy",
                "pistols",
            ):
                continue
            family, group, side = name.split("_")[0], definition["groups"][j], name[-1]
            cc = [
                i
                for i, nm in enumerate(names)
                if definition["semantic"][i] == family
                and definition["groups"][i] == group
                and nm.endswith("_" + side)
                and "_coverage_" not in nm
            ]
            rich_numeric[j] = np.isfinite(rich_numeric[cc]).mean() if cc else 0
        original_inputs[uid] = np.concatenate(
            (numeric, composition, dynamic_features[uid])
        ).astype("float32")
        rich_inputs[uid] = np.concatenate(
            (rich_numeric, composition, dynamic_features[uid])
        ).astype("float32")
        for package in method["parent_model_packages"]:
            if package["task"] == uid[0]:
                assert package["feature_names"] == names + [
                    "composition_" + str(i) for i in range(len(composition))
                ] + [
                    "dynamic_" + name
                    for name in (
                        "log_odds",
                        "team_difference",
                        "player_difference",
                        "map_difference",
                        "agent_context_difference",
                        "series_difference",
                        "coverage_a",
                        "coverage_b",
                        "exposure_a",
                        "exposure_b",
                        "age_a",
                        "age_b",
                        "roster_change_a",
                        "roster_change_b",
                    )
                ]
        audits.append(
            {
                "task": uid[0],
                "game_id": uid[1],
                "prediction_cutoff": cutoff,
                "base_history_end_timestamp": base.last,
                "rich_history_end_timestamp": online.last,
                "base_training_maps": base_cursor,
                "rich_admitted_maps": cursor,
                "new_feedback": False,
            }
        )
    args.output.mkdir(parents=True)
    order = sorted(contexts)
    np.savez_compressed(
        args.output / "features.npz",
        original=np.stack([original_inputs[u] for u in order]),
        rich=np.stack([rich_inputs[u] for u in order]),
    )
    for name, rows_out in [
        ("keys.json", order),
        ("history-audit.json", audits),
        (
            "source-missingness.json",
            {"rounds": reasons, "numeric_conflict_count": len(conflicts)},
        ),
    ]:
        (args.output / name).write_text(json.dumps(rows_out, indent=2) + "\n")
    receipt = {
        "status": "retained_residual_features",
        "complete_cohort": context["status"]
        == "complete_real_source_context_projection"
        and reference.get("complete_cohort", False),
        "rows": len(order),
        "method_sha256": sha(args.method),
        "context_projection_sha256": sha(args.context / "projection.json"),
        "reference_sha256": sha(args.reference / "frozen.json"),
        "code_sha256": sha(__file__),
        "legacy_history_code_sha256": sha(old_path),
        "old_source_inventory_sha256": sha(dynamic / "online/sources.json"),
        "prepared_sha256": sha(prepared_path),
        "parent_prepared_sha256": sha(parent_prepared_path),
        "source_database_sha256": cfg["source_database_sha256"],
        "performance_database_sha256": cfg["database_sha256"],
        "agent_registry_sha256": sha(alias_path),
        "new_targets_or_performance_read": False,
        "fits_or_scores": False,
        "new_feedback": False,
        "runtime": sys.version,
        "files": {p.name: sha(p) for p in args.output.iterdir()},
    }
    (args.output / "frozen.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
