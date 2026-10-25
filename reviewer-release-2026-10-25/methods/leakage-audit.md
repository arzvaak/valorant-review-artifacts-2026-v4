# Bounded implementation and chronology audit

This audit covers the active saved core snapshots and their original immutable code. It is not an independent end-to-end reconstruction, a test suite, or proof of real publication timestamps. The audit does not establish robustness to longer release delays or alternative prediction cutoffs.

| Boundary | Evidence and inspection | Finding / limitation |
|---|---|---|
| Canonical identity and evaluation alignment | All 12 block/task key tables; saved selector outputs | Team orientation and exact keys retained; forecast lengths and finite values checked |
| Event and Match partitioning | Every pair of phases in each block/task | No shared event or Match IDs across phases; later-window reuse as history remains intentional |
| Feature history cutoff | Every saved phase key; source history/player loops | Recorded nonnull history end is strictly before prediction cutoff; history admits training rows only |
| Earlier decision labels | Completion +2h versus next nonempty phase cutoff | All inspected original phase boundaries clear the proxy; minimum clearance 4.916667h |
| Source performance projection | `legacy_prepare.load_observations`, `round_projection`, `player_snapshot.prepare_players` | Performance queries are restricted to training Game IDs; target-map lineup identity can be read for the after-picks task, not current-map performance |
| Series roster/agent inputs | `History.features`, `player_snapshot.preferred` | Earlier unambiguous roster state and most-observed earlier agent; actual future roster/picks absent from series predictors |
| Equal timestamps | `History.update` and cursor conditions | Frozen pre-batch expectations and strict source release; scores/IDs do not establish within-batch map order |
| Preprocessing and category fitting | `preprocess.transform` / `arrays` | Medians, masks, centering, scales and map vocabulary fit training rows; mirrored augmentation remains inside training |
| Model configurations and stopping | `fit.package`, `models.candidate`, tree/neural adapters | Selection scores use selection labels; stopping uses stopping labels; each selected nonlinear configuration has three recorded seeds |
| Operational choice | `selector.select_and_freeze` | Ranking and mixtures use selection events, never evaluation outcomes |
| Forecast calibration | `fit.calibrated` | Separate calibration labels, scalar candidate policy and explicit sparse fallback; evaluation reliability fits are reporting-only |
| Measured live availability | No historical publication clocks recovered | Not established. Archived lineup presence, schedule proxies and strict code inequalities do not prove real-time publication eligibility |

Machine-readable original-snapshot findings are in `core-audit.json`, with phase counts and coverage CSVs. Source/code/input hashes bind this inspection. Original source functions are preserved with the software snapshot. No tests or synthetic rehearsals were added or run. No reserved or locked-confirmatory data was read.

## Separate completed sensitivity execution

The original-snapshot inspection above is distinct from the completed advancing-history and two-clock execution. Supplement S22 records the release guards, shared Match cutoffs, simultaneous updates, six excluded unavailable historical outcomes, retained scoring masks, phase purge, and pre-scoring family freezes. The final-round qualification verifies all 48 policy/block forecast freezes and 10,448 listed payload hashes under their retained cloud/local identities. This is runtime/artifact evidence, not actual publication-time validation or independent pipeline replication. The historical refresh8 activation receipt is unavailable; its original results/completion hashes still match.
