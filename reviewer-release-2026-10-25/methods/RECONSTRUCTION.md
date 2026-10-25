# Reconstructing the reported evidence

The current [v4.2 software and aggregate companion](https://anonymous.4open.science/r/valorant-temporal-benchmark-2026/reviewer-release-2026-10-25/README.md) contains the shared software, methods and aggregate evidence. The earlier hosted v4.1 snapshot remains preserved separately. Neither hosted release contains the manuscript or supplement PDFs.

## Primary probability losses and fixed-decision accuracy

The new companion contains `aggregates/aggregate-reconstruction/event-forecast-losses.csv`: one procedure/event record with row count, sum of Brier losses, sum of natural-log losses and number of correct fixed decisions. Opaque cluster labels preserve pairing within an analysis/task and omit source event/team/player IDs. No row-level probabilities or labels are present. These are saved-forecast sufficient summaries for the stated scores, not a raw forecasting dataset.

Run from the `reviewer-release-2026-10-25/` directory, with Python 3 and no additional packages:

```sh
python software/aggregate-renderers/reconstruct_event_scores.py \
  --input aggregates/aggregate-reconstruction/event-forecast-losses.csv \
  --output reconstructed-scores.csv
```

For each procedure, divide its total loss by total rows for the row-weighted score. For the event-equal score, divide loss by rows within each event and average those event means. Fixed-decision accuracy uses the analogous correct counts. Core decisions use the strict probability >0.5 rule, so ties select canonical Team B. This command neither refits a model nor estimates a new confidence interval. Calibration curves, ROC-AUC and reporting-only diagnostic fits cannot be reconstructed from these grouped sums alone.

The companion `aggregates/aggregate-reconstruction/reconstructed-scores.csv` records the executed reconstruction. Original core selector and K20 map Brier values round to 0.235585 and 0.246114. `aggregates/aggregate-reconstruction/core-coverage.csv` and `aggregates/aggregate-reconstruction/core-phase-counts.csv` are directly readable aggregate tables; `aggregates/aggregate-reconstruction/coverage.tex` and `aggregates/aggregate-reconstruction/phase-counts.tex` are their article presentations.

## Original figures and complete original comparison tables

The carried-forward original archive provides its pinned `figure-reconstruction/` environment and aggregate inputs. From the repository root, enter that directory before running its command:

```sh
docker compose run --build --rm render
```

This reconstructs the original aggregate CSVs and figure sets. It does not fit forecasting models, reread source outcomes or independently reproduce scores. The package README names the carried-forward original archive and the separately added new companion; do not substitute a current remote dataset for the pinned source snapshot.

## Documents and access

The editable TeX, BibTeX, adjacent input tables and figure assets reproduce the article PDF using the supplied class/style and the package's build instructions. The Markdown supplement uses the bundled graphics. The data card explains populations, missingness, availability and redistribution scope. No reconstruction command here accesses protected confirmation, runs a repository test suite or establishes a prospective prediction timestamp.

## Completed sensitivity aggregates

`aggregates/final-sensitivities/all-88-contrasts.csv` contains every declared new contrast, pointwise interval, Monte Carlo quantities and separate-family adjusted decision. `aggregates/final-sensitivities/procedure-metrics.csv` separately labels event-equal and row-weighted losses. Its p >0.5 accuracy is a diagnostic, not the original residual decision thresholds. `aggregates/final-sensitivities/core-ties-by-block.csv`, `aggregates/final-sensitivities/calibration-candidates.csv`, `aggregates/final-sensitivities/schedule-gaps.csv` and `aggregates/final-sensitivities/removal-parent-metrics.csv` expose the bounded saved-output audits. These summaries permit inspection and figure reconstruction, not fresh bootstrap inference or independent row-level refitting. `aggregates/final-sensitivities/specification.json` binds the figure inputs to the saved-report hashes.

The report-to-document renderer is `software/audit-renderers/integrate_final_round.py` in the hosted v4.2 companion. It reads already-scored private report JSONs, verifies result/completion hashes, emits aggregate CSV/Markdown and draws the figure with Matplotlib; it does not read row-level forecasts or fit models. Re-execution requires the private reports at their recorded local layout or a lawful equivalent supplied with matching identities. The distributed aggregate inputs and figure are available without those private files. The separately retained v4.1 snapshot is unchanged.

From the `reviewer-release-2026-10-25/` directory, reconstruct the new figure from shared aggregates alone (Python 3 with Matplotlib):

```sh
python software/aggregate-renderers/plot_final_sensitivities.py --input aggregates/final-sensitivities/all-88-contrasts.csv --output reconstructed-final-figure
```

This plots recorded estimates and intervals; it neither recalculates inference nor accesses private reports or source rows.

## Reader-facing saved-output additions

`aggregates/reader-evidence/later-exact-ties.csv` gives all six original later-cohort procedures with actual tied-subset class counts and original/half-credit row/event decision summaries. It does not expose source IDs and changes no probabilities or loss findings. `aggregates/reader-evidence/online-rating-display.csv` supplies all seven procedures in the expanded online probability-loss figure. Reconstruct that figure using `python software/aggregate-renderers/plot_rating_display.py --input aggregates/reader-evidence/online-rating-display.csv --output reconstructed-rating-figure` with the supplied plotting environment. `aggregates/reader-evidence/saved-output-provenance.json` binds the aggregate sources and diagnostic implementation by hash. No new inference or model fitting is performed.

The optional `software/audit-renderers/presentation_saved_outputs.py` audit renderer requires explicit `--later`, `--aggregates` and `--output` paths. `--later` refers to restricted frozen forecasts/targets, so this optional command is not runnable from shared aggregates alone. Reviewer score/figure reconstruction above does not need it. No author-local path is embedded in this renderer.
