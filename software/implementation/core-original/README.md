# Strong predictor study execution

Owner authorized 2026-10-09. Run `bootstrap.py` once into a new release root,
then `run.py --root <root>` under the isolated CUDA environment. Docker optional;
CUDA mandatory. One active fitting process. No tests or synthetic rehearsals.

The runner durably saves candidates, checkpoints, family packages and whole-stage
freezes. Restart the same command to recover the same input/method/code/seed
identity. Changed result-producing code requires a new versioned release.
Preparation failures are preserved and require an explicit versioned repair.

Inspect `progress.json`, `blocks/*/progress.json`, orchestration logs,
candidate `epoch-telemetry.jsonl`, failures and `core-progress.json`.
`completion.json` alone denotes all planned stages complete. Scored evidence is
in `report/{core,players,removals}`; editable figures and a separate expanded
manuscript are produced after all three stages. Public Git contains code,
configuration and prose; local release stores data/model artifacts.
