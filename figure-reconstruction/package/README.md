# Aggregate and figure reconstruction package

This package contains pinned plotting code and allowlisted grouped aggregate projections. It excludes source rows, row-level labels or forecasts, prediction arrays, model files, local source/input path maps and identity keys. The input/projection, figure-code and output hashes are recorded in the pathless receipts.

From `figure-reconstruction/` at the candidate root:

```sh
docker compose run --build --rm render
```

The command exports included grouped records to CSV and renders figures and presentation tables. The v3 qualification applies to the unchanged carried-forward `outputs-v3/` tree and its input/source identity. A single v4.1 run in a fresh output directory is documented separately in `../reconstruction-qualification-v4.1.json`. No model fitting, scoring, interval calculation, resampling or outcome-row access occurs.

Project-authored renderer code and aggregate materials are licensed under MIT and CC BY 4.0, respectively, within the scope described in `LICENSES/README.md`. Dependency notices are retained under `notices/`. The owner's sharing authorization is not independent verification of source rights. This aggregate package is not a standalone refit or independent scientific reproduction.
