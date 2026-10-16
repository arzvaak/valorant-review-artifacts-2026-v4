"""One-shot, train-history-only inputs for the nine-family exploratory programme.

Numeric missingness is retained. Source statistics are read only for training
maps; evaluation target projection is a separate operation after forecast freeze.
"""

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

PHASES = ("train", "validation", "calibration", "evaluation")
TASKS = ("map_winner", "map_margin", "map_fraction", "series_winner")
FAMILIES = (
    "opening",
    "stars",
    "economy",
    "pistols",
    "map_side",
    "impact",
    "experience",
    "schedule_stability",
    "matchups",
)
PLAYER_FIELDS = (
    "ACS",
    "ADR",
    "KAST",
    "kills",
    "deaths",
    "assists",
    "first_kills",
    "first_deaths",
    "economy_rating_display",
    "multikill_2",
    "multikill_3",
    "multikill_4",
    "multikill_5",
    "clutch_1v1",
    "clutch_1v2",
    "clutch_1v3",
    "clutch_1v4",
    "clutch_1v5",
    "plants",
    "defuses",
    "headshot_percentage_display",
)
RATE_FIELDS = {
    "kills",
    "deaths",
    "assists",
    "first_kills",
    "first_deaths",
    "multikill_2",
    "multikill_3",
    "multikill_4",
    "multikill_5",
    "clutch_1v1",
    "clutch_1v2",
    "clutch_1v3",
    "clutch_1v4",
    "clutch_1v5",
    "plants",
    "defuses",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from common import save_json as durable_save_json

    durable_save_json(path, value)


def verify_registry(root):
    cfg = json.loads((Path(root) / "config.json").read_text())
    registry = Path(cfg["registry_root"])
    release = json.loads((registry / "release.json").read_text())
    for item in release["sources"].values():
        if digest(item["path"]) != item["sha256"]:
            raise RuntimeError("Physical source changed: " + str(item["path"]))
    for name, sha in release["files"].items():
        if digest(registry / name) != sha:
            raise RuntimeError("Registry file changed: " + name)
    # Configured databases must themselves be among the physically bound sources.
    bound = {str(Path(v["path"]).resolve()) for v in release["sources"].values()}
    for field in ("database_path", "source_database_path"):
        if str(Path(cfg[field]).resolve()) not in bound:
            raise RuntimeError("Unbound configured source: " + field)
    rows = json.loads((registry / "maps.json").read_text())
    if len({r["game_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate registry map identity")
    plan = json.loads((registry / "target-plan.json").read_text())
    if set(plan) != {r["game_id"] for r in rows}:
        raise RuntimeError("Target-plan identity differs from registry")
    for r in rows:
        if int(r["team_a_id"]) >= int(r["team_b_id"]):
            raise RuntimeError("Noncanonical registry orientation")
        if not math.isfinite(r["anchor"]):
            raise RuntimeError("Missing schedule cutoff")
        for e in plan[r["game_id"]]:
            if (
                e["game_id"] != r["game_id"]
                or e["match_id"] != r["match_id"]
                or e["input_identity_sha256"] != r["input_identity_sha256"]
            ):
                raise RuntimeError("Target source-plan identity changed")
    return cfg, release, rows, plan


def adapter(source, rows, plan, root):
    from targets import Targets

    result = Targets(source, rows, Path(root))
    result.selected = plan
    return result


def round_projection(source, target, selected):
    """Project only requested registry IDs; no unbound or holdout row is read."""
    ids = [r["game_id"] for r in selected]
    rounds = collections.defaultdict(list)
    for g, number, winner, raw_sha in source.execute(
        "select game_id,round_number,winner_team_id,raw_sha256 from main.map_rounds "
        "where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        rounds[g].append((number, winner, raw_sha))
    result, reasons = {}, {}
    for r in selected:
        g = r["game_id"]
        seq = rounds[g]
        numbers = [n for n, _, _ in seq]
        if any(not isinstance(n, int) for n in numbers):
            reasons[g] = "invalid_round_number"
            continue
        if not seq or sorted(numbers) != list(range(1, len(seq) + 1)):
            reasons[g] = "missing_or_noncontiguous_round_sequence"
            continue
        if any(t not in (r["team_a_id"], r["team_b_id"]) for _, t, _ in seq):
            reasons[g] = "round_winner_orientation_conflict"
            continue
        bound_parent = {e["parent_sha256"] for e in target.selected[g]}
        if any(h not in bound_parent for _, _, h in seq):
            reasons[g] = "round_sequence_source_binding_conflict"
            continue
        won = sum(t == r["team_a_id"] for _, t, _ in seq)
        expected_counts = []
        for e in target.selected[g]:
            document = target.document(e)
            panel = document.css_first('.vm-stats-game[data-game-id="' + g + '"]')
            if panel is None:
                reasons[g] = "bound_game_panel_missing"
                break
            cells = panel.css(".vm-stats-game-header .team .score")
            tokens = [cell.text(strip=True) for cell in cells]
            if len(tokens) != 2 or not all(
                re.fullmatch(r"\d+", token) for token in tokens
            ):
                reasons[g] = "bound_score_projection_invalid"
                break
            counts = [int(token) for token in tokens]
            index = e["source_teams"].index(r["team_a_id"])
            expected_counts.append((counts[index], counts[1 - index]))
        if g in reasons:
            continue
        if any(pair != (won, len(seq) - won) for pair in expected_counts):
            reasons[g] = "round_sequence_exact_source_score_count_disagreement"
            continue
        margin = 2 * won - len(seq)
        try:
            source_winner = target.label(g)
        except ValueError as error:
            reasons[g] = str(error)
            continue
        if not margin or int(margin > 0) != source_winner:
            reasons[g] = "round_sequence_map_target_disagreement"
            continue
        result[g] = (float(margin), won / len(seq), len(seq))
    return result, reasons


def _project(source, target, registry_rows, task, keys):
    byid = {r["game_id"]: r for r in registry_rows}
    quarantine = []
    values = []
    if task != "series_winner":
        for key in keys:
            r = byid.get(key["game_id"])
            if r is None or any(
                key[f] != r[f]
                for f in (
                    "match_id",
                    "event_id",
                    "team_a_id",
                    "team_b_id",
                    "input_identity_sha256",
                )
            ):
                raise RuntimeError("Map key identity changed before target projection")
    if task in ("map_margin", "map_fraction"):
        selected = [byid[k["game_id"]] for k in keys]
        round_values, reasons = round_projection(source, target, selected)
    inventory = target.inventory
    series_inventory = (
        {
            m: set(games)
            for m, games in source.execute(
                "select match_id,archived_game_ids from core.series where match_id in (select unnest(?))",
                [[k["match_id"] for k in keys]],
            ).fetchall()
        }
        if task == "series_winner"
        else {}
    )
    bymatch = collections.defaultdict(set)
    for r in registry_rows:
        bymatch[r["match_id"]].add(r["game_id"])
    for key in keys:
        try:
            if task == "series_winner":
                m = key["match_id"]
                games = set(key["archived_game_ids"])
                if (
                    key["game_id"] != "series:" + m
                    or games != inventory[m]
                    or games != series_inventory.get(m, set())
                    or games != bymatch[m]
                    or not games
                ):
                    raise ValueError("series_incomplete_exact_archived_map_set")
                for g in games:
                    r = byid[g]
                    if any(
                        r[f] != key[f] for f in ("event_id", "team_a_id", "team_b_id")
                    ):
                        raise ValueError("series_registry_orientation_conflict")
                wins = sum(target.label(g) for g in games)
                if 2 * wins == len(games):
                    raise ValueError("series_tied_map_majority")
                value = float(2 * wins > len(games))
            else:
                g = key["game_id"]
                r = byid[g]
                if any(
                    key[f] != r[f]
                    for f in (
                        "match_id",
                        "event_id",
                        "team_a_id",
                        "team_b_id",
                        "input_identity_sha256",
                    )
                ):
                    raise ValueError("map_key_registry_identity_conflict")
                if task == "map_winner":
                    value = float(target.label(g))
                else:
                    if g in reasons:
                        raise ValueError(reasons[g])
                    value = round_values[g][0 if task == "map_margin" else 1]
            values.append(value)
        except (ValueError, KeyError) as error:
            values.append(float("nan"))
            quarantine.append({"game_id": key["game_id"], "reason": str(error)})
    return np.asarray(values, dtype=np.float32), quarantine


def project_labels(root, task, keys):
    """Called by the scorer only after its whole-programme freeze verification.

    NaNs identify reason-only quarantined labels, preserving frozen row order.
    """
    if task not in TASKS:
        raise ValueError("Unknown task")
    cfg, _, rows, plan = verify_registry(root)
    source = duckdb.connect(cfg["source_database_path"], read_only=True)
    try:
        target = adapter(source, rows, plan, root)
        values, quarantine = _project(source, target, rows, task, keys)
        save_json(Path(root) / "label-quarantine" / (task + ".json"), quarantine)
        save_json(
            Path(root) / "inputs" / task / "evaluation-quarantine.json", quarantine
        )
        return values
    finally:
        source.close()


class Mean:
    """Observed sums/exposures; absent numeric observations stay missing."""

    def __init__(self):
        self.sum = collections.defaultdict(float)
        self.count = collections.defaultdict(float)
        self.maps = 0

    def add(self, fields):
        self.maps += 1
        for f, value in fields.items():
            if isinstance(value, tuple):
                numerator, denominator = value
                if (
                    numerator is not None
                    and denominator
                    and math.isfinite(float(numerator))
                ):
                    self.sum[f] += float(numerator)
                    self.count[f] += float(denominator)
            elif value is not None and math.isfinite(float(value)):
                self.sum[f] += float(value)
                self.count[f] += 1

    def value(self, field):
        return (
            self.sum[field] / self.count[field] if self.count[field] else float("nan")
        )


def finite_mean(values):
    v = [x for x in values if math.isfinite(float(x))]
    return float(np.mean(v)) if v else float("nan")


def ratio(a, b):
    return a / b if b else float("nan")


class Columns:
    def __init__(self):
        self.names = []
        self.context = []
        self.families = {f: [] for f in FAMILIES}
        self.swap = []
        self.sign = []

    def pair(self, family, name, a, b, values):
        j = len(self.names)
        self.names.extend([name + "_a", name + "_b"])
        columns = self.context if family == "context" else self.families[family]
        columns.extend([j, j + 1])
        self.swap.extend([j + 1, j])
        self.sign.extend([1, 1])
        values.extend([a, b])

    def odd(self, family, name, value, values):
        j = len(self.names)
        self.names.append(name)
        (self.context if family == "context" else self.families[family]).append(j)
        self.swap.append(j)
        self.sign.append(-1)
        values.append(value)


class History:
    def __init__(self, agents, maps):
        self.agents, self.maps = agents, maps
        self.player = collections.defaultdict(Mean)
        self.player_context = collections.defaultdict(Mean)
        self.recent = collections.defaultdict(collections.deque)
        self.recent_cache = {}
        self.team = collections.defaultdict(Mean)
        self.team_map = collections.defaultdict(Mean)
        self.rating = collections.defaultdict(lambda: 1500.0)
        self.mapwins = collections.defaultdict(lambda: [0.0, 0.0])
        self.usage = collections.defaultdict(collections.Counter)
        self.pairs = collections.Counter()
        self.rosters = {}
        self.schedules = collections.defaultdict(list)
        self.duels = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        self.last = None

    def update(self, batch, lineups, observations, teamobs, duels, labels, reveal):
        delta = collections.defaultdict(float)
        batch_rosters = collections.defaultdict(set)
        series_seen = set()
        # Every expectation is from the same frozen pre-batch K20 state.
        for r in batch:
            g, ta, tb = r["game_id"], r["team_a_id"], r["team_b_id"]
            y = labels[g]
            expected = 1 / (1 + 10 ** ((self.rating[tb] - self.rating[ta]) / 400))
            delta[ta] += 20 * (y - expected)
            delta[tb] -= 20 * (y - expected)
            for t, won, pwin in ((ta, y, expected), (tb, 1 - y, 1 - expected)):
                self.mapwins[t, r["map_entity_id"]][0] += won
                self.mapwins[t, r["map_entity_id"]][1] += 1
                obs = dict(teamobs.get((g, t), {}))
                if math.isfinite(obs.get("margin", float("nan"))):
                    obs["K20_proxy_adjusted_margin"] = (
                        obs["margin"] - (2 * pwin - 1) * obs["rounds"]
                    )
                self.team[t].add(obs)
                self.team_map[t, r["map_entity_id"]].add(obs)
                lineup = [(p, a.lower()) for z, p, a in lineups[g] if z == t]
                players = frozenset(p for p, _ in lineup)
                batch_rosters[t].add(players)
                for p1, p2 in itertools.combinations(sorted(players), 2):
                    self.pairs[t, p1, p2] += 1
                for p, a in lineup:
                    values = observations.get((g, t, p), {})
                    self.player[p].add(values)
                    self.player_context[p, r["map_entity_id"], a].add(values)
                    self.recent[p].append(
                        (r["anchor"] + reveal, r["map_entity_id"], a, values)
                    )
                    self.usage[t]["agent:" + a] += 1
                self.usage[t][r["map_entity_id"]] += 1
                if (r["match_id"], t) not in series_seen:
                    self.schedules[t].append(r["anchor"])
                    series_seen.add((r["match_id"], t))
            for kind, p, q, n, reverse in duels.get(g, []):
                if n is None or reverse is None:
                    continue
                self.duels[kind, p, q][0] += n
                self.duels[kind, p, q][1] += reverse
                self.duels[kind, p, q][2] += 1
                self.duels[kind, q, p][0] += reverse
                self.duels[kind, q, p][1] += n
                self.duels[kind, q, p][2] += 1
        for t, d in delta.items():
            self.rating[t] += d
        for t, rosters in batch_rosters.items():
            # Equal-time disagreement must not be broken by ID or source order.
            self.rosters[t] = (
                next(iter(rosters)) if len(rosters) == 1 else None,
                batch[0]["anchor"] + reveal,
            )
        self.last = batch[0]["anchor"] + reveal
        self.recent_cache.clear()

    def features(self, r, lineups, cutoff, series):
        ta, tb = r["team_a_id"], r["team_b_id"]
        teams = (ta, tb)
        slots = []
        for t in teams:
            if series:
                roster, _ = self.rosters.get(t, (None, None))
                slots.append([(p, None) for p in sorted(roster)] if roster else [])
            else:
                slots.append(
                    sorted(
                        [(p, a.lower()) for z, p, a in lineups[r["game_id"]] if z == t]
                    )
                )
        c, values = Columns(), []
        profiles = [[self.player[p] for p, _ in team] for team in slots]
        contextual = [
            [
                (
                    self.player[p]
                    if series
                    else self.player_context[p, r["map_entity_id"], a]
                )
                for p, a in team
            ]
            for team in slots
        ]
        teamstates = [
            self.team[t] if series else self.team_map[t, r["map_entity_id"]]
            for t in teams
        ]
        c.odd(
            "context",
            "elo_log_odds_difference",
            (self.rating[ta] - self.rating[tb]) * math.log(10) / 400,
            values,
        )
        odds = []
        for t in teams:
            w, n = self.mapwins[t, r.get("map_entity_id")]
            odds.append(math.log((w + 5) / (n - w + 5)) if not series else 0.0)
        c.odd("context", "map_history_log_odds_difference", odds[0] - odds[1], values)
        for name, vv in [
            (
                "profile_coverage",
                [ratio(sum(z.maps > 0 for z in p), 5) for p in profiles],
            ),
            (
                "profile_log_maps",
                [finite_mean([math.log1p(z.maps) for z in p]) for p in profiles],
            ),
            ("roster_proxy_missing", [float(not bool(team)) for team in slots]),
        ]:
            c.pair("context", name, *vv, values)

        def player_mean(field, states=profiles):
            return [finite_mean([p.value(field) for p in side]) for side in states]

        def player_add(family, field, name=None, states=profiles):
            c.pair(family, name or field, *player_mean(field, states), values)

        def team_add(family, field):
            c.pair(family, field, *[s.value(field) for s in teamstates], values)

        for field in ("first_kills", "first_deaths", "opening_balance"):
            player_add("opening", field)
            player_add("opening", field, "matched_" + field, contextual)
        for metric in ("ACS", "ADR", "kills", "KAST"):
            for operation in ("mean", "strongest", "second", "spread", "top_two_gap"):
                v = []
                for side in profiles:
                    nums = sorted(
                        [
                            p.value(metric)
                            for p in side
                            if math.isfinite(p.value(metric))
                        ],
                        reverse=True,
                    )
                    if not nums:
                        value = float("nan")
                    elif operation == "mean":
                        value = float(np.mean(nums))
                    elif operation == "strongest":
                        value = nums[0]
                    elif operation == "second":
                        value = nums[1] if len(nums) > 1 else float("nan")
                    elif operation == "spread":
                        value = float(np.std(nums))
                    else:
                        value = nums[0] - nums[1] if len(nums) > 1 else float("nan")
                    v.append(value)
                c.pair("stars", metric + "_" + operation, *v, values)
        player_add("stars", "kill_share")
        player_add("stars", "kill_share_squared")
        for relation in ("lower", "equal", "higher"):
            team_add("economy", relation + "_equipment_win_fraction")
            team_add("economy", relation + "_equipment_rounds")
        player_add("economy", "economy_rating_display")
        for category in ("eco", "semi_eco", "semi_buy", "full_buy"):
            team_add("economy", category + "_source_buy_win_fraction")
            team_add("economy", category + "_source_buy_rounds")
        for field in (
            "pistol_win_fraction",
            "pistol_opportunities",
            "pistol_win_followup",
            "pistol_loss_followup",
        ):
            team_add("pistols", field)
        for field in (
            "margin",
            "round_fraction",
            "K20_proxy_adjusted_margin",
            "attack_fraction",
            "defence_fraction",
        ):
            team_add("map_side", field)
        for field in (
            "ACS",
            "ADR",
            "KAST",
            "kills",
            "deaths",
            "assists",
            "first_kills",
            "first_deaths",
        ):
            for side in ("t", "ct"):
                player_add("map_side", side + "_" + field)
        for field in (
            "multikill_2",
            "multikill_3",
            "multikill_4",
            "multikill_5",
            "clutch_1v1",
            "clutch_1v2",
            "clutch_1v3",
            "clutch_1v4",
            "clutch_1v5",
            "plants",
            "defuses",
            "assists",
            "headshot_percentage_display",
        ):
            player_add("impact", field)
        experience = []
        for t, side, matched in zip(teams, slots, contextual):
            pairs = [
                self.pairs[t, p, q]
                for p, q in itertools.combinations(sorted(p for p, _ in side), 2)
            ]
            experience.append(
                (
                    finite_mean(pairs),
                    min(pairs) if pairs else float("nan"),
                    finite_mean([p.maps for p in matched]),
                )
            )
        for j, field in enumerate(
            (
                "shared_pair_maps_mean",
                "shared_pair_maps_min",
                "matched_player_map_agent_maps",
            )
        ):
            c.pair("experience", field, experience[0][j], experience[1][j], values)
        # Recent form is a fixed 90-day window; variability/trend preserve the original player/map/agent records.
        recent_stats = []
        for side in slots:
            items = []
            for p, agent in side:
                cachekey = (p, cutoff, None if series else r["map_entity_id"], agent)
                if cachekey not in self.recent_cache:
                    prior = [
                        z
                        for z in self.recent[p]
                        if z[0] > cutoff - 90 * 86400 and z[0] < cutoff
                    ]
                    matched = (
                        prior
                        if series
                        else [
                            z
                            for z in prior
                            if z[1] == r["map_entity_id"] and z[2] == agent
                        ]
                    )
                    state, matched_state = Mean(), Mean()
                    for z in prior:
                        state.add(z[3])
                    for z in matched:
                        matched_state.add(z[3])
                    points = [
                        (z[0] / 86400, z[3].get("ACS"))
                        for z in prior
                        if z[3].get("ACS") is not None
                    ]
                    acs = [v for _, v in points]
                    sd = float(np.std(acs)) if len(acs) > 1 else float("nan")
                    times = np.asarray([t for t, _ in points])
                    metric = np.asarray(acs)
                    slope = (
                        float(
                            np.sum((times - times.mean()) * (metric - metric.mean()))
                            / np.sum((times - times.mean()) ** 2)
                        )
                        if len(points) > 1 and np.ptp(times) > 0
                        else float("nan")
                    )
                    self.recent_cache[cachekey] = (state, matched_state, sd, slope)
                items.append((p, self.recent_cache[cachekey]))
            recent_stats.append(items)
        for field in ("ACS", "ADR", "KAST", "opening_balance"):
            vv = [
                finite_mean(
                    [
                        item[0].value(field) - self.player[p].value(field)
                        for p, item in side
                    ]
                )
                for side in recent_stats
            ]
            c.pair("schedule_stability", "recent90_minus_long_" + field, *vv, values)
            vv = [
                finite_mean([item[1].value(field) for _, item in side])
                for side in recent_stats
            ]
            c.pair("schedule_stability", "recent90_matched_" + field, *vv, values)
        for index, name in (
            (2, "recent90_ACS_variability"),
            (3, "recent90_ACS_trend_per_day"),
        ):
            c.pair(
                "schedule_stability",
                name,
                *[
                    finite_mean([item[index] for _, item in side])
                    for side in recent_stats
                ],
                values
            )
        for field in (
            "hours_since_previous_schedule",
            "matches_last_7_days",
            "matches_last_30_days",
            "roster_continuity",
        ):
            vv = []
            for t, side in zip(teams, slots):
                times = self.schedules[t]
                roster, _ = self.rosters.get(t, (None, None))
                if field == "hours_since_previous_schedule":
                    v = (cutoff - max(times)) / 3600 if times else float("nan")
                elif field.startswith("matches_last_"):
                    days = 7 if "7_" in field else 30
                    v = sum(time > cutoff - days * 86400 for time in times)
                else:
                    v = (
                        len(set(p for p, _ in side) & roster) / 5
                        if roster and side
                        else float("nan")
                    )
                vv.append(v)
            c.pair("schedule_stability", field, *vv, values)
        for kind in ("all_kills", "op_kills", "first_kills"):
            vv, exposure = [], []
            for own, other in ((slots[0], slots[1]), (slots[1], slots[0])):
                records = [self.duels[kind, p, q] for p, _ in own for q, _ in other]
                wins = sum(z[0] for z in records)
                losses = sum(z[1] for z in records)
                vv.append(ratio(wins, wins + losses))
                exposure.append(sum(z[2] for z in records))
            c.pair(
                "opening" if kind == "first_kills" else "matchups",
                kind + "_prior_pair_fraction",
                *vv,
                values
            )
            c.pair(
                "opening" if kind == "first_kills" else "matchups",
                kind + "_prior_pair_maps",
                *exposure,
                values
            )
        # Family coverage and missing flags are members of their own family.
        for family in FAMILIES:
            cols = c.families[family][:]
            left = [values[j] for j in cols if c.names[j].endswith("_a")]
            right = [values[j] for j in cols if c.names[j].endswith("_b")]
            observed = [
                ratio(sum(math.isfinite(float(v)) for v in s), len(s))
                for s in (left, right)
            ]
            c.pair(family, family + "_numeric_coverage", *observed, values)
            c.pair(
                family,
                family + "_missing_prior_history",
                float(not profiles[0] or not any(p.maps for p in profiles[0])),
                float(not profiles[1] or not any(p.maps for p in profiles[1])),
                values,
            )
        composition = []
        vocabulary = (
            ["unknown_agent"]
            + ["agent:" + a for a in self.agents]
            + ["unknown_map"]
            + self.maps
        )
        for t, team in zip(teams, slots):
            comp = np.zeros(len(vocabulary), dtype=np.float32)
            if series:
                counts = self.usage[t]
                for j, key in enumerate(vocabulary):
                    if key.startswith("agent:"):
                        den = sum(
                            v for k, v in counts.items() if k.startswith("agent:")
                        )
                    else:
                        den = sum(v for k, v in counts.items() if k.startswith("map:"))
                    comp[j] = counts[key] / den if den else 0
            else:
                for _, a in team:
                    comp[1 + self.agents.index(a) if a in self.agents else 0] += 1
                name = r["map_entity_id"]
                comp[
                    (
                        len(self.agents) + 2 + self.maps.index(name)
                        if name in self.maps
                        else len(self.agents) + 1
                    )
                ] = 1
            composition.extend(comp)
        return values, composition, c


def load_observations(warehouse, source, trainrows, lineups, valid_rounds):
    valid_round_ids = set(valid_rounds)
    """Bounded bulk SQL, restricted to the exact training Game-ID set."""
    ids = [r["game_id"] for r in trainrows]
    canonical_a = {r["game_id"]: r["team_a_id"] for r in trainrows}
    observed = collections.defaultdict(dict)
    missing_cell_reasons = []
    # Corrected long cells retain explicit nulls; duplicate disagreement fails closed.
    cells = warehouse.execute(
        "select game_id,team_id,player_id,field,value from corrected.player_map_statistics "
        "where game_id in (select unnest(?)) and field in (select unnest(?))",
        [ids, list(PLAYER_FIELDS)],
    ).fetchall()
    unique = collections.defaultdict(set)
    for g, t, p, f, v in cells:
        unique[g, t, p, f].add(None if v is None else float(v))
    for (g, t, p, f), vv in unique.items():
        nonmissing = {v for v in vv if v is not None}
        observed[g, t, p][f] = next(iter(nonmissing)) if len(nonmissing) == 1 else None
        if len(nonmissing) > 1:
            missing_cell_reasons.append(
                {
                    "game_id": g,
                    "team_id": t,
                    "player_id": p,
                    "field": f,
                    "reason": "conflicting_corrected_numeric_cells",
                }
            )
    # Side cells come from the corrected extraction, not current-map snapshots.
    side_cells = collections.defaultdict(set)
    for g, t, p, side, f, v in warehouse.execute(
        "select game_id,team_id,player_id,side,field,value from corrected.player_side_statistics "
        "where game_id in (select unnest(?)) and field in ('ACS','ADR','KAST','kills','deaths','assists','first_kills','first_deaths')",
        [ids],
    ).fetchall():
        if side not in ("t", "ct"):
            raise RuntimeError("Unknown historical side literal")
        side_cells[g, t, p, side, f].add(None if v is None else float(v))
    for (g, t, p, side, f), vv in side_cells.items():
        nonmissing = {v for v in vv if v is not None and math.isfinite(v)}
        observed[g, t, p][side + "_" + f] = (
            next(iter(nonmissing)) if len(nonmissing) == 1 else None
        )
        if len(nonmissing) > 1:
            missing_cell_reasons.append(
                {
                    "game_id": g,
                    "team_id": t,
                    "player_id": p,
                    "field": side + "_" + f,
                    "reason": "conflicting_corrected_side_numeric_cells",
                }
            )
    summaries = {}
    for row in source.execute(
        "select game_id,team_id,observed_rounds,rounds_won,observed_round_margin,attack_rounds,attack_rounds_won,"
        "defence_rounds,defence_rounds_won,pistol_opportunities,pistol_wins "
        "from derived.team_map_round_summary where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        g, t, n, w, margin, attack, aw, defence, dw, pistols, pw = row
        if g not in valid_round_ids:
            continue
        canonical_margin, canonical_fraction, canonical_rounds = valid_rounds[g]
        ta = canonical_a[g]
        expected_margin = canonical_margin if t == ta else -canonical_margin
        expected_fraction = canonical_fraction if t == ta else 1 - canonical_fraction
        if (
            n != canonical_rounds
            or margin != expected_margin
            or abs(float(w) / n - expected_fraction) > 1e-12
        ):
            raise RuntimeError(
                "Historical derived round summary disagrees with source-bound rounds"
            )
        summaries[g, t] = {
            "rounds": float(n),
            "margin": float(margin),
            "round_fraction": (float(w), float(n)),
            "attack_fraction": (float(aw or 0), float(attack or 0)),
            "defence_fraction": (float(dw or 0), float(defence or 0)),
            "attack_rounds": float(attack or 0),
            "defence_rounds": float(defence or 0),
            "pistol_win_fraction": (float(pw or 0), float(pistols or 0)),
            "pistol_opportunities": float(pistols or 0),
        }
    relations = source.execute(
        "select game_id,team_id,equipment_relation,count(*),sum(observed_round_won) "
        "from derived.team_round_observations where game_id in (select unnest(?)) "
        "group by game_id,team_id,equipment_relation",
        [ids],
    ).fetchall()
    for g, t, relation, n, wins in relations:
        if g not in valid_round_ids:
            continue
        if relation not in ("lower_equipment", "equal_equipment", "higher_equipment"):
            continue
        field = relation.split("_")[0]
        summaries.setdefault((g, t), {})[field + "_equipment_win_fraction"] = (
            float(wins),
            float(n),
        )
        summaries[g, t][field + "_equipment_rounds"] = float(n)
    for g, t, won, n, wins in source.execute(
        "select game_id,team_id,pistol_won,count(next_round_won),sum(next_round_won) "
        "from derived.pistol_followup_observations where game_id in (select unnest(?)) "
        "group by game_id,team_id,pistol_won",
        [ids],
    ).fetchall():
        if g not in valid_round_ids:
            continue
        summaries.setdefault((g, t), {})[
            "pistol_win_followup" if won else "pistol_loss_followup"
        ] = (float(wins or 0), float(n))
    buy_labels = {
        "Eco (won)": "eco",
        "$ (won)": "semi_eco",
        "$$ (won)": "semi_buy",
        "$$$ (won)": "full_buy",
    }
    for g, t, label, n, w in warehouse.execute(
        "select game_id,team_id,source_buy_label,round_count_observed,round_wins_observed "
        "from corrected.team_buy_summary where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        if label == "Pistol Won":
            continue  # Success count without an attempt denominator is not a probability.
        if label not in buy_labels:
            raise RuntimeError("Unknown source buy category")
        category = buy_labels[label]
        if n is not None and w is not None and n > 0 and 0 <= w <= n:
            summaries.setdefault((g, t), {})[category + "_source_buy_win_fraction"] = (
                float(w),
                float(n),
            )
            summaries[g, t][category + "_source_buy_rounds"] = float(n)
    for r in trainrows:
        g = r["game_id"]
        for t in (r["team_a_id"], r["team_b_id"]):
            players = [p for z, p, _ in lineups[g] if z == t]
            kills = [observed[g, t, p].get("kills") for p in players]
            team_kills = (
                sum(kills)
                if len(kills) == 5
                and all(v is not None and math.isfinite(v) for v in kills)
                else None
            )
            rounds = summaries.get((g, t), {}).get("rounds")
            for p in players:
                v = observed[g, t, p]
                if team_kills and v.get("kills") is not None:
                    v["kill_share"] = v["kills"] / team_kills
                    v["kill_share_squared"] = v["kill_share"] ** 2
                fk, fd = v.get("first_kills"), v.get("first_deaths")
                v["opening_balance"] = (
                    (fk - fd, rounds)
                    if rounds and fk is not None and fd is not None
                    else None
                )
                for side, denominator in (
                    ("t", summaries.get((g, t), {}).get("attack_rounds")),
                    ("ct", summaries.get((g, t), {}).get("defence_rounds")),
                ):
                    for f in (
                        "kills",
                        "deaths",
                        "assists",
                        "first_kills",
                        "first_deaths",
                    ):
                        numerator = v.get(side + "_" + f)
                        v[side + "_" + f] = (
                            (numerator, denominator)
                            if numerator is not None and denominator
                            else None
                        )
                for f in RATE_FIELDS:
                    numerator = v.get(f)
                    v[f] = (
                        (numerator, rounds)
                        if numerator is not None and rounds
                        else None
                    )
    duels = collections.defaultdict(list)
    seen = {}
    for g, kind, p, q, n, reverse in warehouse.execute(
        "select game_id,matrix_kind,row_player_id,column_player_id,row_to_column_count,column_to_row_count "
        "from corrected.player_opponent_matrices where game_id in (select unnest(?))",
        [ids],
    ).fetchall():
        # Source tables contain both orientations; accept each unordered cell once.
        k = (g, kind, min(p, q), max(p, q))
        oriented = (n, reverse) if p < q else (reverse, n)
        if k in seen and seen[k] != oriented:
            raise RuntimeError("Historical matchup duplicate disagreement")
        seen[k] = oriented
    for (g, kind, p, q), (n, reverse) in seen.items():
        if kind not in ("all_kills", "Op_Kills", "first_kills"):
            raise RuntimeError("Unreviewed matchup matrix kind: " + str(kind))
        duels[g].append(
            (
                {
                    "all_kills": "all_kills",
                    "Op_Kills": "op_kills",
                    "first_kills": "first_kills",
                }[kind],
                p,
                q,
                n,
                reverse,
            )
        )
    return observed, summaries, duels, missing_cell_reasons


def main(root, repository):
    root = Path(root)
    out = root / "inputs"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            "Preparation is one-shot; preserve unfinished attempt and version repair"
        )
    out.mkdir(parents=True, exist_ok=True)
    cfg, release, allrows, plan = verify_registry(root)
    partition = json.loads((root / "partition.json").read_text())
    split = partition["splits"]
    if any(v not in PHASES for v in split.values()):
        raise RuntimeError("Unknown partition phase")
    source = duckdb.connect(cfg["source_database_path"], read_only=True)
    warehouse = duckdb.connect(cfg["database_path"], read_only=True)
    source.execute("set threads=4")
    warehouse.execute("set threads=4")
    try:
        target = adapter(source, allrows, plan, root)
        quarantine = []
        anchors = collections.defaultdict(set)
        for r in allrows:
            anchors[r["match_id"]].add(r["anchor"])
        ambiguous = {m for m, times in anchors.items() if len(times) != 1}
        rows = []
        for row in allrows:
            if row["event_id"] not in split:
                continue
            if row["match_id"] in ambiguous:
                quarantine.append(
                    {
                        "game_id": row["game_id"],
                        "reason": "ambiguous_series_schedule_anchor",
                    }
                )
                continue
            r = dict(row)
            r["phase"] = split[r["event_id"]]
            if r["region"] == "global" and r["phase"] != "evaluation":
                raise RuntimeError("Known global event is evaluation-only")
            rows.append(r)
        rows.sort(key=lambda r: (r["anchor"], r["game_id"]))
        if not rows:
            raise RuntimeError("Empty real block")
        matchphases = collections.defaultdict(set)
        for r in rows:
            matchphases[r["match_id"]].add(r["phase"])
        if any(len(v) != 1 for v in matchphases.values()):
            raise RuntimeError("Series crosses partition")
        for left, right in zip(PHASES, PHASES[1:]):
            earlier = [r for r in rows if r["phase"] == left]
            later = [r for r in rows if r["phase"] == right]
            if (
                earlier
                and later
                and max(r["event_completion"] for r in earlier)
                >= min(r["anchor"] for r in later)
            ):
                raise RuntimeError("Event calendar extents overlap split")
        ids = [r["game_id"] for r in rows]
        lineups = collections.defaultdict(list)
        for g, t, p, a in source.execute(
            "select game_id,team_id,player_id,agent_alias from core.lineups "
            "where game_id in (select unnest(?))",
            [ids],
        ).fetchall():
            if not a:
                raise RuntimeError("Missing registry lineup agent")
            lineups[g].append((t, p, a))
        for r in rows:
            for t in (r["team_a_id"], r["team_b_id"]):
                team = [z for z in lineups[r["game_id"]] if z[0] == t]
                if (
                    len(team) != 5
                    or len({z[1] for z in team}) != 5
                    or len({z[2].lower() for z in team}) != 5
                ):
                    raise RuntimeError("Registry strict lineup changed")
        trainrows = [r for r in rows if r["phase"] == "train"]
        # Invalid source targets remain reason-only evidence, never observations
        # in ratings, performance histories, roster proxies or vocabularies.
        history_labels, invalid_history_targets = _project(
            source, target, allrows, "map_winner", trainrows
        )
        quarantine.extend(
            dict(
                item,
                task="map_winner",
                phase="train",
                scope="historical_invalid_target_excluded",
            )
            for item in invalid_history_targets
        )
        training_labels = {
            r["game_id"]: float(label)
            for r, label in zip(trainrows, history_labels)
            if np.isfinite(label)
        }
        trainrows = [r for r in trainrows if r["game_id"] in training_labels]
        if not set(training_labels.values()) <= {0, 1}:
            raise RuntimeError("Invalid training map winner")
        agents = sorted(
            {a.lower() for r in trainrows for _, _, a in lineups[r["game_id"]]}
        )
        maps = sorted({r["map_entity_id"] for r in trainrows})
        valid_rounds, invalid_rounds = round_projection(source, target, trainrows)
        quarantine.extend(
            {
                "game_id": g,
                "reason": reason,
                "scope": "historical_round_features_missing",
            }
            for g, reason in invalid_rounds.items()
        )
        observed, teamobs, duels, missing_cell_reasons = load_observations(
            warehouse, source, trainrows, lineups, valid_rounds
        )
        reveal = float(cfg.get("reveal_minutes", 120)) * 60
        lead = float(cfg.get("lead_minutes", 15)) * 60
        if reveal != 7200 or lead != 900:
            raise RuntimeError(
                "Availability/prediction proxy changed from fixed method"
            )
        grouped = collections.defaultdict(list)
        all_by_match = collections.defaultdict(set)
        for r in allrows:
            all_by_match[r["match_id"]].add(r["game_id"])
        for r in rows:
            grouped[r["match_id"]].append(r)
        seriesrows = []
        for m, group in grouped.items():
            games = {r["game_id"] for r in group}
            if games != target.inventory[m] or games != all_by_match[m]:
                quarantine.append(
                    {
                        "game_id": "series:" + m,
                        "reason": "series_incomplete_exact_archived_map_set",
                    }
                )
                continue
            if (
                len(
                    {
                        (
                            r["event_id"],
                            r["team_a_id"],
                            r["team_b_id"],
                            r["anchor"],
                            r["phase"],
                        )
                        for r in group
                    }
                )
                != 1
            ):
                quarantine.append(
                    {
                        "game_id": "series:" + m,
                        "reason": "series_registry_identity_disagreement",
                    }
                )
                continue
            k = {
                f: group[0][f]
                for f in (
                    "match_id",
                    "event_id",
                    "team_a_id",
                    "team_b_id",
                    "anchor",
                    "event_completion",
                    "phase",
                    "region",
                    "population",
                    "group",
                    "classification_state",
                )
            }
            k.update(
                game_id="series:" + m,
                archived_game_ids=sorted(games),
                roster_basis="last_unambiguous_strictly_prior_training_appearance_proxy",
                actual_format_used_as_predictor=False,
            )
            seriesrows.append(k)
        # One stream of predictions, and one strictly earlier equal-time update cursor.
        stream = sorted(
            [(r, False) for r in rows] + [(r, True) for r in seriesrows],
            key=lambda z: (z[0]["anchor"], z[0]["game_id"]),
        )
        history = History(agents, maps)
        cursor = 0
        records = {"map": [], "series": []}
        audits = []
        metadata = {}
        for ordinal, (r, series) in enumerate(stream):
            cutoff = r["anchor"] - lead
            while (
                cursor < len(trainrows)
                and trainrows[cursor]["anchor"] + reveal < cutoff
            ):
                end = cursor + 1
                while (
                    end < len(trainrows)
                    and trainrows[end]["anchor"] == trainrows[cursor]["anchor"]
                ):
                    end += 1
                batch = trainrows[cursor:end]
                if any(z["match_id"] == r["match_id"] for z in batch):
                    raise RuntimeError("Target series entered history")
                history.update(
                    batch, lineups, observed, teamobs, duels, training_labels, reveal
                )
                cursor = end
            if history.last is not None and history.last >= cutoff:
                raise RuntimeError("Strict history cutoff violated")
            numeric, composition, columns = history.features(r, lineups, cutoff, series)
            kind = "series" if series else "map"
            key = dict(r)
            key.update(
                prediction_cutoff=cutoff,
                prediction_cutoff_utc=datetime.fromtimestamp(
                    cutoff, timezone.utc
                ).isoformat(),
                history_end_timestamp=history.last,
                source_availability="schedule_plus_120_minutes_proxy",
            )
            records[kind].append((key, numeric, composition))
            current = {
                "context_columns": columns.context,
                "family_columns": columns.families,
                "feature_names": columns.names,
                "composition_width": len(composition),
                "swap_numeric_indices": columns.swap,
                "swap_numeric_signs": columns.sign,
                "swap_composition_indices": list(
                    range(len(composition) // 2, len(composition))
                )
                + list(range(len(composition) // 2)),
                "swap_composition_signs": [1] * len(composition),
            }
            if kind in metadata and metadata[kind] != current:
                raise RuntimeError("Feature definition changed within block")
            metadata[kind] = current
            audits.append(
                {
                    "game_id": key["game_id"],
                    "phase": key["phase"],
                    "prediction_cutoff": cutoff,
                    "history_end_timestamp": history.last,
                    "training_only": True,
                    "equal_time_frozen_batch": True,
                    "target_series_excluded": True,
                }
            )
            if ordinal % 2000 == 0:
                print("HISTORY", ordinal, "/", len(stream), flush=True)
        taskmeta = {}
        for task in TASKS:
            kind = "series" if task == "series_winner" else "map"
            definition = metadata.get(kind)
            if definition is None:
                # Same declared numeric layout and training vocabularies; empty series
                # phases are explicit data absence, requiring a fitter fallback.
                definition = metadata["map"]
            counts, coverage = {}, {}
            for phase in PHASES:
                selected = [z for z in records[kind] if z[0]["phase"] == phase]
                keys = [z[0] for z in selected]
                labels = None
                if phase != "evaluation":
                    labels, failed = _project(source, target, allrows, task, keys)
                    quarantine.extend(dict(z, task=task, phase=phase) for z in failed)
                    good = np.isfinite(labels)
                    selected = [z for z, keep in zip(selected, good) if keep]
                    keys = [z[0] for z in selected]
                    labels = labels[good]
                folder = out / task
                folder.mkdir(exist_ok=True)
                x = np.asarray([z[1] for z in selected], dtype=np.float32).reshape(
                    -1, len(definition["feature_names"])
                )
                comp = np.asarray([z[2] for z in selected], dtype=np.float32).reshape(
                    -1, definition["composition_width"]
                )
                if np.isinf(x).any() or not np.isfinite(comp).all():
                    raise RuntimeError("Invalid numeric/composition snapshot")
                np.savez_compressed(
                    folder / (phase + ".npz"), numeric=x, composition=comp
                )
                save_json(folder / (phase + "-keys.json"), keys)
                if labels is not None:
                    np.save(folder / (phase + "-labels.npy"), labels)
                counts[phase] = {
                    "rows": len(keys),
                    "maps": (
                        len(keys)
                        if kind == "map"
                        else sum(len(k["archived_game_ids"]) for k in keys)
                    ),
                    "events": len({k["event_id"] for k in keys}),
                    "series": len({k["match_id"] for k in keys}),
                }
                coverage[phase] = {
                    family: {
                        "numeric_cells": int(x[:, cols].size),
                        "missing_cells": int(np.isnan(x[:, cols]).sum()),
                    }
                    for family, cols in definition["family_columns"].items()
                }
            taskmeta[task] = dict(
                definition,
                counts=counts,
                coverage=coverage,
                label_scale=(
                    "original_fraction_0_1"
                    if task == "map_fraction"
                    else (
                        "raw_round_margin"
                        if task == "map_margin"
                        else "canonical_team_a_binary"
                    )
                ),
                series_roster_proxy=kind == "series",
            )
        save_json(out / "quarantine.json", quarantine)
        save_json(out / "numeric-cell-missingness.json", missing_cell_reasons)
        save_json(out / "history-audit.json", audits)
        save_json(
            out / "identity-audit.json",
            {
                "sources": release["sources"],
                "registry_maps": len(allrows),
                "retained_maps": len(rows),
                "series_identity_candidates": len(seriesrows),
                "splits": split,
                "field_minimal_population": True,
                "numeric_missingness_retained": True,
                "evaluation_labels_read": False,
                "protected_data_read": False,
            },
        )
        # Commit every snapshot payload to storage before the durable receipt.
        for payload in out.rglob("*"):
            if payload.is_file():
                with payload.open("rb") as stream:
                    os.fsync(stream.fileno())
        for directory in [out] + [p for p in out.rglob("*") if p.is_dir()]:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        files = {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        prepared = {
            "files": files,
            "sources": release["sources"],
            "registry_release_sha256": digest(
                Path(cfg["registry_root"]) / "release.json"
            ),
            "config_sha256": digest(root / "config.json"),
            "partition_sha256": digest(root / "partition.json"),
            "code_sha256": digest(__file__),
            "tasks": taskmeta,
            "families": list(FAMILIES),
            "agent_vocabulary_training_only": agents,
            "map_vocabulary_training_only": maps,
            "chronology_audit": {
                "rows": len(audits),
                "history_strictly_before_prediction": True,
                "equal_time_frozen_batch": True,
                "training_only": True,
                "source_reveal_minutes": 120,
                "prediction_lead_minutes": 15,
            },
            "evaluation_labels_read": False,
            "numeric_missingness_retained": True,
            "rate_aggregation": "ratio_of_observed_success_or_count_sums_to_observed_exposure_sums",
            "K20_proxy_adjusted_margin": "fixed match-win expectation transformed to round-margin proxy; not calibrated expected round margin",
            "timing_proof": "retrospective schedule availability proxy; no prospective publication timing proof",
            "operator_kills_matrix": "Op_Kills is Operator weapon kills, not opening kills",
            "source_buy_categories": {
                "eco": "Eco (won)",
                "semi_eco": "$ (won)",
                "semi_buy": "$$ (won)",
                "full_buy": "$$$ (won)",
            },
            "quarantine_rows": len(quarantine),
        }
        save_json(out / "prepared.json", prepared)
        save_json(root / "prepared.json", prepared)
        print(
            "PREPARED",
            {task: meta["counts"] for task, meta in taskmeta.items()},
            flush=True,
        )
    finally:
        source.close()
        warehouse.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.repository)
