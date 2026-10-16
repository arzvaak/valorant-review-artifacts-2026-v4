# Revision source inventory — 2026-10-16 scientific release

`current-live-source/` holds the current scientific analysis helpers and their direct `later_score.py` dependency. It includes the corrected future feedback, player-input, representation code, retuned-removal and secondary-metric methods. `SOURCE-IDENTITIES.json` distinguishes historical execution identities; `CURRENT-LIVE-SOURCE-MANIFEST.json` binds the current helper set and its path-only artifact-root adapter.

The older `execution-sources/`, `historical-code-snapshots/`, and `repaired-future-source/` directories are retained as provenance. They distinguish original execution code from later repairs; do not conflate them.

No phase inputs, row-level labels/forecasts, prediction arrays, matrices, checkpoints, identity keys, model binaries, or source captures accompany this code. Grouped results are projected separately under `figure-reconstruction/package/data/revision/`. Retuning or independently recomputing results requires compatible phase inputs and a CUDA-enabled, fully matched environment, neither included nor qualified.

Project-authored software is under MIT as scoped in the candidate root `LICENSES/README.md`. The included Set Transformer license applies only to that upstream contribution. Third-party terms and notices remain in `notices/`.
