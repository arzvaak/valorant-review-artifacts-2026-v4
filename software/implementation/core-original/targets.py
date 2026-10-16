"""Source-bound exact-map score projection; no score appears in logs/artifacts."""

import gzip, hashlib, json, re
from collections import OrderedDict, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from selectolax.lexbor import LexborHTMLParser


class Targets:
    def __init__(self, source, rows, root):
        self.rows = {r["game_id"]: r for r in rows}
        self.root = root
        self.cache = OrderedDict()
        self.sources = defaultdict(list)
        self.inventory = defaultdict(set)
        for g, m in source.execute("select game_id,match_id from core.maps").fetchall():
            self.inventory[m].add(g)
        for g, m, p, h, parent, url in source.execute(
            "select distinct game_id,match_id,body_path,body_sha256,parent_raw_sha256,source_url from extracted.archive_sections where family='overview' and game_id in (select unnest(?)) and body_path is not null",
            [list(self.rows)],
        ).fetchall():
            self.sources[g].append(
                {
                    "game_id": g,
                    "match_id": m,
                    "path": p,
                    "body_sha256": h,
                    "parent_sha256": parent,
                    "url": url,
                }
            )
        self.selected = {}
        self.failures = []

    def document(self, entry):
        h = entry["body_sha256"]
        if h in self.cache:
            self.cache.move_to_end(h)
            return self.cache[h]
        b = gzip.decompress(Path(entry["path"]).read_bytes())
        if hashlib.sha256(b).hexdigest() != h:
            raise ValueError("source_body_hash_changed")
        d = LexborHTMLParser(b.decode("utf8", errors="replace"))
        self.cache[h] = d
        if len(self.cache) > 16:
            self.cache.popitem(last=False)
        return d

    def preflight(self, g):
        r = self.rows[g]
        accepted = []
        reasons = []
        for e in self.sources[g]:
            try:
                if e["match_id"] != r["match_id"] or not re.match(
                    r"^/" + re.escape(r["match_id"]) + r"(?:/|$)",
                    urlparse(e["url"]).path,
                ):
                    raise ValueError("source_match_id_conflict")
                d = self.document(e)
                header = d.css_first(".match-header")
                event = header.css_first("a.match-header-event") if header else None
                if not event or not re.match(
                    r"^/event/" + re.escape(r["event_id"]) + r"(?:/|$)",
                    event.attributes.get("href", ""),
                ):
                    raise ValueError("source_event_id_conflict")
                sides = []
                for side in ["mod-1", "mod-2"]:
                    links = header.css("a.match-header-link." + side)
                    if len(links) != 1:
                        raise ValueError("source_side_identity_cardinality")
                    m = re.match(
                        r"^/team/(\d+)(?:/|$)", links[0].attributes.get("href", "")
                    )
                    if not m:
                        raise ValueError("source_team_id_missing")
                    sides.append(m[1])
                if set(sides) != {r["team_a_id"], r["team_b_id"]}:
                    raise ValueError("source_team_identity_conflict")
                games = {
                    n.attributes.get("data-game-id")
                    for n in d.css(".vm-stats-game[data-game-id]")
                } - {None, "all"}
                if games != self.inventory[r["match_id"]]:
                    raise ValueError("complete_source_game_inventory_conflict")
                panels = d.css('.vm-stats-game[data-game-id="' + g + '"]')
                if len(panels) != 1:
                    raise ValueError("exact_game_panel_cardinality")
                panel = panels[0]
                name = panel.css_first(".vm-stats-game-header .map > div > span")
                if not name:
                    raise ValueError("source_map_name_missing")
                observed = " ".join(name.text(deep=False, strip=True).split())
                if observed.casefold() != r["map_name"].casefold():
                    raise ValueError("source_map_name_conflict")
                teams = panel.css(".vm-stats-game-header .team")
                if (
                    len(teams) != 2
                    or "mod-right" in teams[0].attributes.get("class", "")
                    or "mod-right" not in teams[1].attributes.get("class", "")
                ):
                    raise ValueError("map_score_side_layout_conflict")
                if any(len(t.css(".score")) != 1 for t in teams):
                    raise ValueError("map_score_cell_cardinality")
                # Published match completion status is a gate; score content remains unread.
                notes = [
                    " ".join(n.text(separator=" ", strip=True).split()).lower()
                    for n in header.css(".match-header-vs-note")
                ]
                if not any(n in ("final", "completed") for n in notes):
                    raise ValueError("source_match_not_final")
                accepted.append(
                    {
                        **e,
                        "source_teams": sides,
                        "input_identity_sha256": r["input_identity_sha256"],
                    }
                )
            except (ValueError, OSError) as error:
                reasons.append(str(error))
        if not accepted:
            self.failures.append(
                {
                    "game_id": g,
                    "reason": sorted(set(reasons)) or ["missing_bound_overview_source"],
                }
            )
            return False
        # All accepted source copies will be checked for label agreement at projection.
        self.selected[g] = accepted
        return True

    def label(self, g):
        if g not in self.selected:
            raise RuntimeError("No source identity freeze before target projection")
        r = self.rows[g]
        labels = []
        for e in self.selected[g]:
            d = self.document(e)
            p = d.css_first('.vm-stats-game[data-game-id="' + g + '"]')
            teams = p.css(".vm-stats-game-header .team")
            values = []
            for t in teams:
                token = t.css_first(".score").text(strip=True)
                if not re.fullmatch(r"\d+", token):
                    raise ValueError("quarantine_noninteger_map_score")
                values.append(int(token))
            if values[0] == values[1]:
                raise ValueError("quarantine_tied_map_score")
            winner = e["source_teams"][0 if values[0] > values[1] else 1]
            labels.append(int(winner == r["team_a_id"]))
        if len(set(labels)) != 1:
            raise ValueError("quarantine_conflicting_map_scores")
        return labels[0]
