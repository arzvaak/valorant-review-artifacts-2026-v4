"""Export included aggregate JSON records to flat CSV tables without rescoring."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "revision"
OUT = ROOT / "rendered" / "aggregate-tables" / "revision"
FORBIDDEN = {
    "game_id", "match_id", "player_id", "team_id", "predictions", "features",
    "labels", "target_rows", "raw_rows",
}
CONTEXT_FIELDS = {
    "task", "block", "scope", "package", "metric", "a", "b", "regime",
    "mode", "model", "event_id", "population", "region", "reference",
    "arm", "status", "mirror",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix="", output=None):
    output = {} if output is None else output
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN:
                raise RuntimeError("Row-level or protected field found in aggregate input")
            flatten(child, f"{prefix}.{key}" if prefix else key, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    else:
        output[prefix] = value
    return output


def safe_stem(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    schema = {}
    inputs = []

    def walk(value, path, context):
        if isinstance(value, dict):
            local = dict(context)
            for key, child in value.items():
                if key.lower() in FORBIDDEN:
                    raise RuntimeError("Row-level or protected field found in aggregate input")
                if key in CONTEXT_FIELDS and not isinstance(child, (dict, list)):
                    local[key] = child
            for key, child in value.items():
                child_path = f"{path}/{safe_stem(key)}"
                if isinstance(child, list) and child and all(isinstance(row, dict) for row in child):
                    table = tables.setdefault(child_path, [])
                    for row in child:
                        flat = dict(local)
                        flat.update(flatten(row))
                        table.append(flat)
                    for row in child:
                        walk(row, child_path + "/nested", local)
                else:
                    walk(child, child_path, local)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child, path + "/item", context)

    files = sorted(DATA.rglob("*.json"))
    for path in files:
        if path.name == "projection-receipt.json":
            continue
        rel = path.relative_to(DATA).as_posix()
        value = json.loads(path.read_text())
        walk(value, safe_stem(rel.removesuffix(".json")), {})
        inputs.append({"path": rel, "sha256": sha(path)})

    outputs = []
    for table_path, rows in sorted(tables.items()):
        name = safe_stem(table_path) + ".csv"
        dest = OUT / name
        keys = sorted({key for row in rows for key in row})
        with dest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        outputs.append({"path": dest.relative_to(ROOT / "rendered").as_posix(), "sha256": sha(dest), "aggregate_records": len(rows)})
        schema[dest.relative_to(ROOT / "rendered").as_posix()] = keys

    schema_path = ROOT / "rendered" / "schemas" / "revision-aggregate-layout.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    outputs.append({"path": schema_path.relative_to(ROOT / "rendered").as_posix(), "sha256": sha(schema_path), "aggregate_records": None})
    manifest = {"inputs": inputs, "outputs": outputs, "rule": "CSV records are aggregate groups already present in included JSON; no row-level observations, metric computation, or resampling is performed."}
    manifest_path = ROOT / "rendered" / "aggregate-tables" / "revision" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Aggregate tables exported: {len(tables)} CSV tables; schema at rendered/schemas/revision-aggregate-layout.json")


if __name__ == "__main__":
    main()
