"""Strict-prior player sets and series preferred-agent profiles from real history."""

import collections, math
from pathlib import Path
import duckdb, numpy as np
import legacy_prepare as old
from common import *

PLAYER_FIELDS = list(old.PLAYER_FIELDS) + [
    s + "_" + f
    for s in ("t", "ct")
    for f in (
        "ACS",
        "ADR",
        "KAST",
        "kills",
        "deaths",
        "assists",
        "headshot_percentage_display",
        "first_kills",
        "first_deaths",
    )
]


def merge(states):
    m = old.Mean()
    for s in states:
        m.maps += s.maps
        for f, v in s.sum.items():
            m.sum[f] += v
        for f, v in s.count.items():
            m.count[f] += v
    return m


def preferred(history, p):
    contexts = [(a, s) for a, s in history.player_agent[p].items() if s.maps]
    counts = collections.Counter()
    for a, s in contexts:
        counts[a] += s.maps
    agent = min(counts, key=lambda a: (-counts[a], a)) if counts else None
    return agent, merge([s for a, s in contexts if a == agent])


def prepare_players(root, parent, receipt, meta):
    cfg = read(parent / "config.json")
    registry = read(Path(cfg["registry_root"]) / "maps.json")
    plan = read(Path(cfg["registry_root"]) / "target-plan.json")
    byid = {k["game_id"]: k for k in registry}
    trainkeys = read(parent / "inputs/map_winner/train-keys.json")
    labels = np.load(parent / "inputs/map_winner/train-labels.npy", allow_pickle=False)
    trainrows = [byid[k["game_id"]] for k in trainkeys]
    labelmap = {k["game_id"]: float(y) for k, y in zip(trainkeys, labels)}
    keys = {
        t: {p: read(root / "inputs" / t / f"{p}-keys.json") for p in PHASES}
        for t in TASKS
    }
    ids = set(k["game_id"] for kk in keys["map_winner"].values() for k in kk) | set(
        labelmap
    )
    source = duckdb.connect(cfg["source_database_path"], read_only=True)
    warehouse = duckdb.connect(cfg["database_path"], read_only=True)
    source.execute("set threads=4")
    warehouse.execute("set threads=4")
    try:
        lineups = collections.defaultdict(list)
        for g, t, p, a in source.execute(
            "select game_id,team_id,player_id,agent_alias from core.lineups where game_id in (select unnest(?))",
            [sorted(ids)],
        ).fetchall():
            lineups[g].append((t, p, a))
        target = old.adapter(source, registry, plan, root)
        rounds, reasons = old.round_projection(source, target, trainrows)
        observed, teamobs, duels, missing = old.load_observations(
            warehouse, source, trainrows, lineups, rounds
        )
        history = old.History(
            receipt["agent_vocabulary_training_only"],
            receipt["map_vocabulary_training_only"],
        )
        history.player_agent = collections.defaultdict(
            lambda: collections.defaultdict(old.Mean)
        )
        trainrows.sort(key=lambda k: (k["anchor"], k["game_id"]))
        cursor = 0
        stream = []
        players = {}
        agents = {}
        series_link = {}
        audit = []
        for task in TASKS:
            for phase in PHASES:
                players[task, phase] = np.full(
                    (len(keys[task][phase]), 2, 5, len(PLAYER_FIELDS) * 3 + 1),
                    np.nan,
                    dtype="float32",
                )
                agents[task, phase] = np.zeros(
                    (len(keys[task][phase]), 2, 5), dtype="int64"
                )
                for i, k in enumerate(keys[task][phase]):
                    stream.append((k["prediction_cutoff"], task, phase, i, k))
        agent_vocab = {
            a: i + 1 for i, a in enumerate(receipt["agent_vocabulary_training_only"])
        }
        cache = {}
        for ordinal, (cutoff, task, phase, i, k) in enumerate(
            sorted(stream, key=lambda z: (z[0], z[4]["game_id"], z[1]))
        ):
            while (
                cursor < len(trainrows) and trainrows[cursor]["anchor"] + 7200 < cutoff
            ):
                end = cursor + 1
                while (
                    end < len(trainrows)
                    and trainrows[end]["anchor"] == trainrows[cursor]["anchor"]
                ):
                    end += 1
                batch = trainrows[cursor:end]
                if any(r["match_id"] == k["match_id"] for r in batch):
                    raise RuntimeError("Target series entered player history")
                history.update(batch, lineups, observed, teamobs, duels, labelmap, 7200)
                for br in batch:
                    for bt, bp, ba in lineups[br["game_id"]]:
                        history.player_agent[bp][ba.lower()].add(
                            observed.get((br["game_id"], bt, bp), {})
                        )
                cursor = end
                cache.clear()
            if history.last is not None and history.last >= cutoff:
                raise RuntimeError("Player history cutoff failure")
            if history.last != k["history_end_timestamp"]:
                raise RuntimeError(
                    "Replayed player history differs from inherited history end"
                )
            rows = []
            for side, t in enumerate((k["team_a_id"], k["team_b_id"])):
                if task == "map_winner":
                    slots = sorted(
                        (p, a.lower()) for z, p, a in lineups[k["game_id"]] if z == t
                    )
                else:
                    roster, _ = history.rosters.get(t, (None, None))
                    slots = [(p, None) for p in sorted(roster)] if roster else []
                for slot, (p, actualagent) in enumerate(slots):
                    ck = (p, actualagent, k.get("map_entity_id"), cutoff)
                    if ck not in cache:
                        if task == "series_winner":
                            a, linked = preferred(history, p)
                        else:
                            a = actualagent
                            linked = history.player_context[p, k["map_entity_id"], a]
                        recent = old.Mean()
                        recentlinked = old.Mean()
                        for timestamp, mp, ag, values in history.recent[p]:
                            if cutoff - 90 * 86400 < timestamp < cutoff:
                                recent.add(values)
                                if ag == a and (
                                    task == "series_winner" or mp == k["map_entity_id"]
                                ):
                                    recentlinked.add(values)
                        vector = [
                            s.value(f)
                            for s in (history.player[p], recent, linked)
                            for f in PLAYER_FIELDS
                        ] + [math.log1p(history.player[p].maps)]
                        cache[ck] = (a, vector, linked, recentlinked)
                    a, vector, linked, recentlinked = cache[ck]
                    players[task, phase][i, side, slot] = vector
                    agents[task, phase][i, side, slot] = agent_vocab.get(a, 0)
                    if task == "series_winner":
                        rows.append((side, linked, recentlinked))
            if task == "series_winner":
                replacements = {}
                for field in ("first_kills", "first_deaths", "opening_balance"):
                    for side, suffix in enumerate(("a", "b")):
                        replacements["matched_" + field + "_" + suffix] = (
                            old.finite_mean(
                                [l.value(field) for z, l, rr in rows if z == side]
                            )
                        )
                for side, suffix in enumerate(("a", "b")):
                    replacements["matched_player_map_agent_maps_" + suffix] = (
                        old.finite_mean([l.maps for z, l, rr in rows if z == side])
                    )
                    for field in ("ACS", "ADR", "KAST", "opening_balance"):
                        replacements["recent90_matched_" + field + "_" + suffix] = (
                            old.finite_mean(
                                [rr.value(field) for z, l, rr in rows if z == side]
                            )
                        )
                series_link[phase, i] = replacements
            audit.append(
                {
                    "task": task,
                    "phase": phase,
                    "game_id": k["game_id"],
                    "history_end_timestamp": history.last,
                    "prediction_cutoff": cutoff,
                    "training_only": True,
                }
            )
            if ordinal % 5000 == 0:
                print(
                    "PLAYER HISTORY", root.name, ordinal, "/", len(stream), flush=True
                )
        for task in TASKS:
            for phase in PHASES:
                save_npz(
                    root / "inputs" / task / f"{phase}-players.npz",
                    players=players[task, phase],
                    agents=agents[task, phase],
                )
                if task == "series_winner":
                    path = root / "inputs" / task / f"{phase}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        x = z["numeric"].copy()
                        comp = z["composition"].copy()
                    m = meta[task]
                    for i in range(len(x)):
                        for n, v in series_link[phase, i].items():
                            x[i, m["feature_names"].index(n)] = v
                    for fam in SEMANTIC:
                        for suffix in ("a", "b"):
                            cols = [
                                j
                                for j, (n, g, f) in enumerate(
                                    zip(m["feature_names"], m["groups"], m["semantic"])
                                )
                                if g == "L"
                                and f == fam
                                and n.endswith("_" + suffix)
                                and "_coverage_" not in n
                            ]
                            n = fam + "_L_coverage_" + suffix
                            if n in m["feature_names"]:
                                x[:, m["feature_names"].index(n)] = (
                                    np.isfinite(x[:, cols]).mean(1) if cols else 0
                                )
                    save_npz(path, numeric=x, composition=comp)
            meta[task].update(
                player_fields=PLAYER_FIELDS,
                player_width=len(PLAYER_FIELDS) * 3 + 1,
                player_agent_vocabulary=agent_vocab,
                series_linked_method=(
                    "most-observed-prior-agent per player, summed histories across historical maps; tie uses agent literal"
                    if task == "series_winner"
                    else "chosen-map/chosen-agent strictly-prior profile"
                ),
            )
        save_json(root / "inputs/player-history-audit.json", audit)
        save_json(
            root / "inputs/player-source-missingness.json",
            {
                "invalid_training_rounds": reasons,
                "missing_cell_count": len(missing),
                "source_projection": "training-only exact Game IDs",
                "evaluation_labels_read": False,
            },
        )
    finally:
        source.close()
        warehouse.close()
