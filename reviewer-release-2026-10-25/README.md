# VALORANT history benchmark: software and aggregate companion (25 October 2026)

This dated companion supports the matching double-anonymous review materials. It adds completed aggregate evidence and current execution source to the older v4.1 software/aggregate release. The v4.1 files remain unchanged. This repository contains no manuscript or supplement PDF.

## What is included

- The nine current retrospective execution modules in `software/final_colab/`, plus aggregate-only score/figure renderers and source-only audit renderers.
- Aggregate event-loss sufficient summaries, reconstructed score tables, 88 completed sensitivity contrasts, procedure metrics, calibration candidates, support/removal/schedule summaries, core coverage and phase counts, and selected audit records.
- Figures and small aggregate tables that expose the seven online references and the later-cohort tied-decision diagnostics.
- Methods, a data card, leakage/availability audit, a finding-to-evidence map, reconstruction limits, and exact recorded execution environments.
- A per-file SHA-256 manifest and package qualification receipt.

The original v4.1 release remains available in the root paths, including the pinned source software, original schemas and feature dictionaries, third-party notices, and aggregate reconstruction environment. Start with [the original artifact README](../../README.md), [`schemas/`](../../schemas/), [`software/revision-20261016/`](../../software/revision-20261016/), and [`figure-reconstruction/`](../../figure-reconstruction/).

## Reconstruct supplied aggregate scores

From this dated release directory, using Python 3:

```sh
python software/aggregate-renderers/reconstruct_event_scores.py \
  --input aggregates/aggregate-reconstruction/event-forecast-losses.csv \
  --output reconstructed-scores.csv
```

This standard-library command reconstructs row-weighted and event-equal Brier/log loss and fixed-decision accuracy from the included grouped sums. It does not rescore row-level predictions or calculate uncertainty.

## Reconstruct supplied figures

With Matplotlib installed:

```sh
python software/aggregate-renderers/plot_final_sensitivities.py \
  --input aggregates/final-sensitivities/all-88-contrasts.csv \
  --output reconstructed-history-figure

python software/aggregate-renderers/plot_rating_display.py \
  --input aggregates/reader-evidence/online-rating-display.csv \
  --output reconstructed-rating-figure
```

These commands plot already-reported estimates and intervals; they do not recalculate inference. To reproduce the original figures, use the pinned aggregate-only renderer in the repository's root `figure-reconstruction/` directory.

## Limits and access boundary

Shared aggregates permit inspection and the reconstruction described above. They are not a row-level training/evaluation dataset and cannot independently regenerate every feature, refit the forecast models, or validate historical publication times. Raw records, source IDs, row-level targets and forecasts, fitted model binaries, checkpoints, credentials, and protected confirmation are not included. Full model refitting requires compatible inputs obtained lawfully through a separate arrangement.

The new sensitivity estimates include saved fits from two recorded environments; see [`methods/METHODS-AND-EXECUTION.md`](methods/METHODS-AND-EXECUTION.md) and [`environments/runtime-record.json`](environments/runtime-record.json). These are mixed-environment sensitivity results, not hardware-matched replication. Non-rejection is not evidence of equivalence.

## License

The repository's existing MIT license covers project-authored code; CC BY 4.0 applies to authored documentation and aggregate figures/tables only to the extent rights are held. Existing third-party notices continue to apply. License scope does not grant rights to underlying third-party source records.
