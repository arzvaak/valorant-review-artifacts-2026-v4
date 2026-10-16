"""Real-source context projection. Does not project targets/performance fields."""

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import os
import re
import sys
import yaml
from selectolax.parser import HTMLParser


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_id(href, kind):
    match = re.match(r"^/" + kind + r"/(\d+)(?:/|$)", href)
    if not match:
        raise ValueError("missing_or_invalid_" + kind + "_identity")
    return match[1]


def direct_context(body):
    html = HTMLParser(body)
    teams = []
    for side in ("mod-1", "mod-2"):
        node = html.css_first(".match-header-link." + side)
        teams.append(source_id(node.attributes.get("href", "") if node else "", "team"))
    maps = []
    for game in html.css("div.vm-stats-game"):
        gid = game.attributes.get("data-game-id", "")
        if gid == "all":
            continue
        if not gid.isdigit():
            raise ValueError("missing_game_identity")
        # Remove named duration/pick nodes without reading their content.
        name = game.css_first(".vm-stats-game-header .map")
        if name:
            for child in name.css(".picked,.pick,.map-duration"):
                child.decompose()
        map_name = (
            " ".join(name.text(separator=" ", strip=True).split()) if name else ""
        )
        slots = []
        lineup_reason = None
        tables = game.css("table.wf-table-inset.mod-overview")
        modern = False
        if not tables:
            tables = game.css(".ovw-table")
            modern = True
        if len(tables) != 2:
            maps.append(
                {
                    "game_id": gid,
                    "map_name": map_name,
                    "lineups": [],
                    "lineup_state": "missing_two_team_overview",
                }
            )
            continue
        for team, table in zip(teams, tables):
            rows = table.css(".ovw-row" if modern else "tbody tr")
            for row in rows:
                links = row.css('a[href*="/player/"]')
                if not links:
                    continue
                if len(links) != 1:
                    lineup_reason = "ambiguous_player_link"
                    break
                if modern:
                    icons = row.css(".ovw-agents img")
                else:
                    cells = row.css("td")
                    icons = cells[1].css("img") if len(cells) > 1 else []
                if len(icons) != 1:
                    lineup_reason = "ambiguous_agent_icon"
                    break
                try:
                    player_id = source_id(links[0].attributes.get("href", ""), "player")
                except ValueError:
                    lineup_reason = "invalid_player_link"
                    break
                slots.append(
                    {
                        "team_id": team,
                        "player_id": player_id,
                        "agent_alias": icons[0].attributes.get("title")
                        or icons[0].attributes.get("alt", ""),
                    }
                )
            if lineup_reason:
                break
        maps.append(
            {
                "game_id": gid,
                "map_name": map_name,
                "lineups": slots,
                "lineup_state": lineup_reason or "identity_projected",
            }
        )
    return teams, maps


def archive_context(body):
    # Decode payload; whitelist only identity/context. Never traverse stat/score fields.
    payload = json.loads(body)["data"]
    teams = [str(team["id"]) for team in payload["teams"]]
    maps = []
    for game in payload["maps"]:
        slots = []
        for team, side in zip(teams, ("team1", "team2")):
            for player in game.get("players", {}).get(side, []):
                slots.append(
                    {
                        "team_id": team,
                        "player_id": str(player.get("id", "")),
                        "agent_alias": player.get("agent", ""),
                    }
                )
        maps.append(
            {
                "game_id": str(game["game_id"]),
                "map_name": game.get("map_name", ""),
                "lineups": slots,
                "lineup_state": "identity_projected",
            }
        )
    return teams, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", type=Path, required=True)
    ap.add_argument("--supplemental-root", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert (
        not args.output.exists()
    ), "Preserve prior partial projection; use a fresh version"
    method = json.loads(args.method.read_text())
    src = Path(os.environ["VALORANT_ARCHIVE_ROOT"]) / "strong-predictors-20261010-v6"
    cfg = json.loads((src / "config.json").read_text())
    registry = Path(cfg["registry_root"])
    maps_path = registry / "maps.json"
    release = json.loads((registry / "release.json").read_text())
    assert sha(maps_path) == release["files"]["maps.json"]
    known_maps = {}
    for row in json.loads(maps_path.read_text()):
        name = row["map_name"].strip().casefold()
        if name in known_maps:
            assert known_maps[name] == row["map_entity_id"]
        known_maps[name] = row["map_entity_id"]
    agents_path = Path("configs/entities/vlr-agents-9e9ec894.yaml").resolve()
    agents = {
        r["observed_alias"].casefold(): r["agent_entity_id"]
        for r in yaml.safe_load(agents_path.read_text())["agents"]
        if r["review_state"] == "reviewed"
    }
    series, maps, quarantine, sources = [], [], [], []
    receipts = sorted((args.capture_root / "captures").glob("*-receipt.json"))
    if args.require_complete:
        for root in (args.capture_root, args.supplemental_root):
            assert (
                root is not None
            ), "Complete projection requires the declared archive-field recovery"
            completion = json.loads((root / "completion.json").read_text())
            job_path = root / "job.json"
            job = json.loads(job_path.read_text())
            assert completion["status"] == "source_capture_complete"
            assert (
                completion["method_sha256"] == job["method_sha256"] == sha(args.method)
            )
            assert completion["job_sha256"] == sha(job_path)
            assert completion["source_plan_sha256"] == job["source_plan_sha256"]
            plan_path = Path(job["source_plan_path"])
            assert sha(plan_path) == job["source_plan_sha256"]
            expected = {
                (r["match_id"], r["event_id"])
                for r in json.loads(plan_path.read_text())["source_plan"]
            }
            actual = {(r["match_id"], r["event_id"]) for r in completion["records"]}
            assert actual == expected and len(actual) == len(completion["records"])
            paths = list((root / "captures").glob("*-receipt.json"))
            assert len(paths) == len(actual)
            for rec in completion["records"]:
                assert (
                    sha(root / "captures" / (rec["match_id"] + "-receipt.json"))
                    == rec["receipt_sha256"]
                )
    supplemental = {}
    if args.supplemental_root:
        job = json.loads((args.supplemental_root / "job.json").read_text())
        assert job["method_sha256"] == sha(args.method)
        plan_path = Path(job["source_plan_path"])
        assert sha(plan_path) == job["source_plan_sha256"]
        plan = json.loads(plan_path.read_text())
        allowed = {str(r["match_id"]): str(r["event_id"]) for r in plan["source_plan"]}
        for path in (args.supplemental_root / "captures").glob("*-receipt.json"):
            replacement = json.loads(path.read_text())
            mid = replacement["match_id"]
            assert mid in allowed and replacement["event_id"] == allowed[mid]
            assert mid not in supplemental and replacement["route"] == "direct_vlr"
            supplemental[mid] = path
        assert set(supplemental) <= {
            json.loads(p.read_text())["match_id"] for p in receipts
        }
    for receipt in receipts:
        rec = json.loads(receipt.read_text())
        assert rec["method_sha256"] == sha(args.method)
        eid, mid = rec["event_id"], rec["match_id"]
        assert (
            eid in method["event_allowlist"]
            and eid not in method["protected_event_denylist"]
        )
        raw = receipt.parent / (
            mid + (".archive.json" if rec["route"] == "archive" else ".html")
        )
        assert sha(raw) == rec["sha256"]
        sources.append(
            {
                "receipt_path": str(receipt),
                "receipt_sha256": sha(receipt),
                "raw_path": str(raw),
                "raw_sha256": rec["sha256"],
            }
        )
        if mid in supplemental:
            assert (
                rec["route"] == "archive"
            ), "Supplement is only for the declared archive field gap"
            replacement_path = supplemental[mid]
            replacement = json.loads(replacement_path.read_text())
            replacement_raw = replacement_path.parent / (mid + ".html")
            assert replacement["method_sha256"] == sha(args.method)
            assert sha(replacement_raw) == replacement["sha256"]

            def utc_clock(token):
                clock = dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
                return (
                    clock.replace(tzinfo=dt.timezone.utc)
                    if clock.tzinfo is None
                    else clock.astimezone(dt.timezone.utc)
                )

            assert utc_clock(
                replacement["metadata"]["scheduled_source_token"]
            ) == utc_clock(rec["metadata"]["scheduled_source_token"])
            old_teams, old_games = archive_context(raw.read_bytes())
            new_teams, new_games = direct_context(replacement_raw.read_bytes())
            assert (
                old_teams == new_teams
            ), "Archive/direct source-team identity conflict"
            assert {g["game_id"] for g in old_games} == {
                g["game_id"] for g in new_games
            }, "Archive/direct map inventory conflict"
            assert {
                g["game_id"]: g["map_name"].strip().casefold() for g in old_games
            } == {
                g["game_id"]: g["map_name"].strip().casefold() for g in new_games
            }, "Archive/direct map identity conflict"
            sources.append(
                {
                    "receipt_path": str(replacement_path),
                    "receipt_sha256": sha(replacement_path),
                    "raw_path": str(replacement_raw),
                    "raw_sha256": replacement["sha256"],
                    "replaces_context_receipt_sha256": sha(receipt),
                    "reason": "declared_empty_archive_player_arrays",
                }
            )
            receipt, rec, raw = replacement_path, replacement, replacement_raw
        try:
            token = rec["metadata"]["scheduled_source_token"]
            # data-utc-ts is published UTC schedule metadata, not actual start proof.
            anchor = dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=dt.timezone.utc)
            endpoint = dt.datetime.fromisoformat(
                method["actual_existing_registry_latest_anchor_utc"].replace(
                    "Z", "+00:00"
                )
            )
            assert anchor > endpoint, "schedule_not_after_actual_history_endpoint"
            teams, games = (
                archive_context if rec["route"] == "archive" else direct_context
            )(raw.read_bytes())
            if (
                len(teams) != 2
                or len(set(teams)) != 2
                or any(not t.isdigit() for t in teams)
            ):
                raise ValueError("invalid_two_team_identity")
            ta, tb = sorted(teams, key=int)
            common = {
                "match_id": mid,
                "event_id": eid,
                "team_a_id": ta,
                "team_b_id": tb,
                "source_team_a_id": teams[0],
                "source_team_b_id": teams[1],
                "orientation_reversed": teams[0] != ta,
                "anchor": anchor.timestamp(),
                "prediction_cutoff": anchor.timestamp() - 900,
                "clock_state": "published_utc_schedule_not_actual_start",
                "region": "unknown",
                "population": "unknown",
                "classification_state": "later_cohort_unreviewed",
                "feedback_allowed": False,
                "source_receipt_sha256": sha(receipt),
                "raw_sha256": rec["sha256"],
            }
            game_ids = [g["game_id"] for g in games]
            if len(game_ids) != len(set(game_ids)) or any(
                not g.isdigit() for g in game_ids
            ):
                raise ValueError("invalid_map_inventory")
            series.append(
                common
                | {
                    "task": "series_winner",
                    "game_id": "series:" + mid,
                    "archived_game_ids": game_ids,
                    "roster_basis": "strictly_prior_appearance_proxy_only",
                    "actual_format_used_as_predictor": False,
                }
            )
            for game in games:
                gid = game["game_id"]
                name = game["map_name"].strip().casefold()
                slots = game["lineups"]
                reason = None
                if game["lineup_state"] != "identity_projected":
                    reason = game["lineup_state"]
                elif name not in known_maps:
                    reason = "unreviewed_map_identity"
                elif len(slots) != 10 or len({s["player_id"] for s in slots}) != 10:
                    reason = "incomplete_unique_5v5_lineup"
                elif any(not s["player_id"].isdigit() for s in slots):
                    reason = "invalid_player_identity"
                elif any(
                    s["agent_alias"].strip().casefold() not in agents for s in slots
                ):
                    reason = "unreviewed_agent_alias"
                else:
                    for team in teams:
                        own = [s for s in slots if s["team_id"] == team]
                        if (
                            len(own) != 5
                            or len(
                                {
                                    agents[s["agent_alias"].strip().casefold()]
                                    for s in own
                                }
                            )
                            != 5
                        ):
                            reason = "incomplete_unique_five_agent_team"
                if reason:
                    quarantine.append(
                        {
                            "match_id": mid,
                            "event_id": eid,
                            "game_id": gid,
                            "reason": reason,
                        }
                    )
                    continue
                maps.append(
                    common
                    | {
                        "task": "map_winner",
                        "game_id": gid,
                        "map_entity_id": known_maps[name],
                        "lineups": [
                            {
                                "team_id": s["team_id"],
                                "player_id": s["player_id"],
                                "agent_entity_id": agents[
                                    s["agent_alias"].strip().casefold()
                                ],
                            }
                            for s in slots
                        ],
                        "composition_availability": "assumed_after_picks_not_publication_clock_verified",
                    }
                )
        except (ValueError, KeyError, TypeError, AssertionError):
            quarantine.append(
                {
                    "match_id": mid,
                    "event_id": eid,
                    "game_id": None,
                    "reason": "source_context_identity_or_clock_unresolved",
                }
            )
    assert len({r["game_id"] for r in maps}) == len(maps)
    assert len({r["game_id"] for r in series}) == len(series)
    args.output.mkdir(parents=True)
    for name, rows in [
        ("map-context.json", maps),
        ("series-context.json", series),
        ("quarantine.json", quarantine),
        ("sources.json", sources),
    ]:
        (args.output / name).write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "status": (
            "complete_real_source_context_projection"
            if args.require_complete
            else "partial_real_source_context_projection"
        ),
        "captured_matches": len(receipts),
        "series_context_rows": len(series),
        "map_context_rows": len(maps),
        "supplemental_context_matches": len(supplemental),
        "quarantine_reasons": dict(
            collections.Counter(r["reason"] for r in quarantine)
        ),
        "targets_or_performance_projected": False,
        "forecasts_or_scores": False,
        "method_sha256": sha(args.method),
        "code_sha256": sha(__file__),
        "agent_registry_sha256": sha(agents_path),
        "old_map_registry_sha256": sha(maps_path),
        "environment": {"python": sys.version},
        "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "projection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
