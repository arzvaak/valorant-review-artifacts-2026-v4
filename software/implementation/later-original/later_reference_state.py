"""Replay admitted old outcomes to later contexts; never read new outcomes.

The numerical rules are those of dynamic-predictor v1 and its matched K20
baseline. New unreviewed-cohort contexts are score-only, so are not enqueued
for feedback. This produces reference features, not a fitted model or score.
"""

import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
from pathlib import Path
import os
import sys

import duckdb
import numpy as np
import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class State:
    def __init__(self, lineups, maps):
        self.lineups, self.maps = lineups, maps
        self.values, self.rosters, self.last = {}, {}, {}
        self.exposure = collections.Counter()
        self.ratings = collections.defaultdict(float)
        self.prior = {t: [0.0, 0.0] for t in ("map_winner", "series_winner")}
        self.history_end = None

    def value(self, key, cutoff):
        mu, when = self.values.get(key, (0.0, cutoff))
        return mu * 2 ** (-(cutoff - when) / (180 * 86400))

    def vector(self, k, task, cutoff):
        teams = (k["team_a_id"], k["team_b_id"])
        if task == "map_winner":
            slots = [
                [(p, a) for t, p, a in self.lineups[k["game_id"]] if t == team]
                for team in teams
            ]
            mp = self.maps[k["game_id"]]
            assert all(len(side) == 5 for side in slots)
        else:
            slots = [
                [(p, None) for p in sorted(self.rosters.get(team) or [])]
                for team in teams
            ]
            mp = None
        mean = lambda vals: float(np.mean(vals)) if vals else 0.0
        td = self.value(("team", teams[0]), cutoff) - self.value(
            ("team", teams[1]), cutoff
        )
        sd = self.value(("series", teams[0]), cutoff) - self.value(
            ("series", teams[1]), cutoff
        )
        pd = mean([self.value(("player", p), cutoff) for p, a in slots[0]]) - mean(
            [self.value(("player", p), cutoff) for p, a in slots[1]]
        )
        md = (
            self.value(("map", teams[0], mp), cutoff)
            - self.value(("map", teams[1], mp), cutoff)
            if mp
            else 0.0
        )
        ad = (
            mean([self.value(("agent", p, mp, a), cutoff) for p, a in slots[0]])
            - mean([self.value(("agent", p, mp, a), cutoff) for p, a in slots[1]])
            if mp
            else 0.0
        )
        z = td + 0.35 * pd + 0.25 * md + 0.10 * ad if mp else sd + 0.5 * td + 0.35 * pd
        coverage = [
            sum(("player", p) in self.values for p, a in side) / 5 for side in slots
        ]
        exposure = [math.log1p(self.exposure[t]) for t in teams]
        age = [
            (cutoff - self.last[t]) / 86400 if t in self.last else float("nan")
            for t in teams
        ]
        change = [
            (
                1 - len(set(p for p, a in side) & (self.rosters.get(t) or set())) / 5
                if side and self.rosters.get(t)
                else float("nan")
            )
            for t, side in zip(teams, slots)
        ]
        vector = np.array(
            [z, td, pd, md, ad, sd, *coverage, *exposure, *age, *change],
            dtype="float32",
        )
        elo = 1 / (
            1
            + np.exp(
                -(self.ratings[task, teams[0]] - self.ratings[task, teams[1]])
                * math.log(10)
                / 400
            )
        )
        return vector, slots, mp, float(elo)

    def release(self, when, batch, metadata, refs, arrays, stored):
        delta, elo_delta = collections.defaultdict(float), collections.defaultdict(
            float
        )
        appeared, seen = collections.defaultdict(set), set()
        valid = 0
        for uid in batch:
            k = metadata[uid]
            if k["region"] == "global":
                continue
            path, index = refs[uid]
            y = float(arrays[path][index])
            if not math.isfinite(y):
                continue
            assert y in (0.0, 1.0)
            vector, slots, mp, elo = stored.pop(uid)
            innovation = y - float(sigmoid(vector[0]))
            valid += 1
            task = uid[0]
            self.prior[task][0] += y
            self.prior[task][1] += 1
            for team, side, sign in zip(
                (k["team_a_id"], k["team_b_id"]), slots, (1.0, -1.0)
            ):
                kind = "team" if task == "map_winner" else "series"
                delta[kind, team] += 0.15 * sign * innovation
                elo_delta[task, team] += 20 * sign * (y - elo)
                if task == "map_winner":
                    delta["map", team, mp] += 0.10 * sign * innovation
                    if len(side) == 5:
                        appeared[team].add(frozenset(p for p, a in side))
                    for p, a in side:
                        delta["player", p] += 0.03 * sign * innovation
                        delta["agent", p, mp, a] += 0.02 * sign * innovation
                    if (team, k["match_id"]) not in seen:
                        self.exposure[team] += 1
                        self.last[team] = when
                        seen.add((team, k["match_id"]))
        for key, d in delta.items():
            self.values[key] = (self.value(key, when) + d, when)
        for key, d in elo_delta.items():
            self.ratings[key] += d
        for team, rosters in appeared.items():
            self.rosters[team] = set(next(iter(rosters))) if len(rosters) == 1 else None
        if valid:
            self.history_end = when
        return valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "Use a fresh immutable reference-feature release"
    method, projection = read(args.method), read(args.context / "projection.json")
    assert projection["method_sha256"] == sha(args.method)
    for f, h in projection["files"].items():
        assert sha(args.context / f) == h
    parent = Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "dynamic-predictor-20261010-v1"
    core = Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "strong-predictors-20261010-v6"
    cfg = read(core / "config.json")
    records = read(parent / "online/sources.json")
    assert (
        sha(parent / "online/sources.json")
        == read(parent / "online/frozen.json")["sources"]
    )
    metadata, refs, arrays = {}, {}, {}
    endpoint = dt.datetime.fromisoformat(
        method["actual_existing_registry_latest_anchor_utc"].replace("Z", "+00:00")
    ).timestamp()
    for rec in records:
        assert (
            sha(rec["keys"]) == rec["keys_sha256"]
            and sha(rec["labels"]) == rec["labels_sha256"]
        )
        arrays[rec["labels"]] = np.load(rec["labels"], mmap_mode="r")
        task = Path(rec["keys"]).parent.name
        for i, k in enumerate(read(rec["keys"])):
            assert k["anchor"] <= endpoint
            assert k["event_id"] not in method["protected_event_denylist"]
            uid = task, k["game_id"]
            if uid in metadata:
                assert all(
                    k[f] == metadata[uid][f]
                    for f in (
                        "match_id",
                        "event_id",
                        "team_a_id",
                        "team_b_id",
                        "anchor",
                        "prediction_cutoff",
                    )
                )
            metadata.setdefault(uid, k)
            refs.setdefault(uid, (rec["labels"], i))
    registry = Path(cfg["registry_root"])
    assert (
        sha(registry / "maps.json")
        == read(registry / "release.json")["files"]["maps.json"]
    )
    maps = {r["game_id"]: r["map_entity_id"] for r in read(registry / "maps.json")}
    ids = [g for t, g in metadata if t == "map_winner"]
    assert sha(cfg["source_database_path"]) == cfg["source_database_sha256"]
    con = duckdb.connect(cfg["source_database_path"], read_only=True)
    lineups = collections.defaultdict(list)
    for g, t, p, a in con.execute(
        "select game_id,team_id,player_id,agent_alias from core.lineups where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        lineups[str(g)].append((str(t), str(p), a.lower()))
    con.close()
    observed_aliases = {a for slots in lineups.values() for t, p, a in slots}
    agent_path = Path("configs/entities/vlr-agents-9e9ec894.yaml").resolve()
    alias_by_entity = collections.defaultdict(set)
    for r in yaml.safe_load(agent_path.read_text())["agents"]:
        if (
            r["review_state"] == "reviewed"
            and r["observed_alias"].lower() in observed_aliases
        ):
            alias_by_entity[r["agent_entity_id"]].add(r["observed_alias"].lower())
    later = {}
    for name in ("map-context.json", "series-context.json"):
        for k in read(args.context / name):
            assert not k[
                "feedback_allowed"
            ], "New feedback admission requires an explicit source-bound policy release"
            assert k["anchor"] > endpoint and k["prediction_cutoff"] > endpoint
            assert k["event_id"] in method["event_allowlist"]
            uid = k["task"], k["game_id"]
            assert uid not in metadata and uid not in later
            later[uid] = k
            if uid[0] == "map_winner":
                assert k["game_id"] not in ids
                maps[k["game_id"]] = k["map_entity_id"]
                for slot in k["lineups"]:
                    aliases = alias_by_entity[slot["agent_entity_id"]]
                    assert (
                        len(aliases) == 1
                    ), "Agent cannot be mapped unambiguously into the retained history basis"
                    lineups[k["game_id"]].append(
                        (slot["team_id"], slot["player_id"], next(iter(aliases)))
                    )
    groups = collections.defaultdict(list)
    for uid, k in metadata.items():
        groups[k["prediction_cutoff"]].append((False, uid))
    for uid, k in later.items():
        groups[k["prediction_cutoff"]].append((True, uid))
    state, pending, stored = State(lineups, maps), [], {}
    features, baseline, audit = {}, {}, []
    counter = 0
    for ordinal, (cutoff, units) in enumerate(sorted(groups.items())):
        releases = collections.defaultdict(list)
        while pending and pending[0][0] < cutoff:
            when, _, uid = heapq.heappop(pending)
            releases[when].append(uid)
        for when, batch in sorted(releases.items()):
            state.release(when, batch, metadata, refs, arrays, stored)
        assert state.history_end is None or state.history_end < cutoff
        for is_later, uid in units:
            k = later[uid] if is_later else metadata[uid]
            assert int(k["team_a_id"]) < int(k["team_b_id"])
            x, slots, mp, elo = state.vector(k, uid[0], cutoff)
            if is_later:
                features[uid] = x
                baseline[uid] = (
                    elo,
                    (state.prior[uid[0]][0] + 5) / (state.prior[uid[0]][1] + 10),
                )
                audit.append(
                    {
                        "task": uid[0],
                        "game_id": uid[1],
                        "history_end_timestamp": state.history_end,
                        "prediction_cutoff": cutoff,
                        "new_outcome_feedback": False,
                    }
                )
            else:
                stored[uid] = (x, slots, mp, elo)
                heapq.heappush(pending, (k["anchor"] + 21600, counter, uid))
                counter += 1
        if ordinal % 5000 == 0:
            print("OLD_HISTORY_REPLAY", ordinal, "/", len(groups), flush=True)
    args.output.mkdir(parents=True)
    order = sorted(features)
    np.savez_compressed(
        args.output / "features.npz",
        dynamic=np.stack([features[u] for u in order]),
        matched_baselines=np.array([baseline[u] for u in order]),
    )
    for f, rows in [("keys.json", order), ("history-audit.json", audit)]:
        (args.output / f).write_text(json.dumps(rows, indent=2) + "\n")
    receipt = {
        "status": "later_reference_features",
        "complete_cohort": projection["status"]
        == "complete_real_source_context_projection",
        "rows": len(order),
        "old_history_units": len(metadata),
        "method_sha256": sha(args.method),
        "context_projection_sha256": sha(args.context / "projection.json"),
        "code_sha256": sha(__file__),
        "old_sources_sha256": sha(parent / "online/sources.json"),
        "source_database_sha256": cfg["source_database_sha256"],
        "agent_registry_sha256": sha(agent_path),
        "new_targets_or_performance_read": False,
        "new_feedback": False,
        "fits_or_scores": False,
        "runtime": sys.version,
        "files": {p.name: sha(p) for p in args.output.iterdir()},
    }
    (args.output / "frozen.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
