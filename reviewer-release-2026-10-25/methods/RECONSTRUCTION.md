# Reconstructing the reported evidence

The original hosted v4.1 snapshot and this dated companion have distinct scopes. The v4.1 snapshot remains unchanged; this versioned companion adds the completed 25 October software and aggregate evidence. Neither anonymous mirror hosts the manuscript or supplement.

## Primary probability losses and fixed-decision accuracy

The new companion contains `aggregate-reconstruction/event-forecast-losses.csv`: one procedure/event record with row count, sum of Brier losses, sum of natural-log losses and number of correct fixed decisions. Opaque cluster labels preserve pairing within an analysis/task and omit source event/team/player IDs. No row-level probabilities or labels are present. These are saved-forecast sufficient summaries for the stated scores, not a raw forecasting dataset.

From the companion root, with Python 3 and no additional packages:

```sh
python software/aggregate-renderers/reconstruct_event_scores.py \
  --input aggregate-reconstruction/event-forecast-losses.csv \
  --output reconstructed-scores.csv
```

For each procedure, divide its total loss by total rows for the row-weighted score. For the event-equal score, divide loss by rows within each event and average those event means. Fixed-decision accuracy uses the analogous correct counts. Core decisions use the strict probability >0.5 rule, so ties select canonical Team B. This command neither refits a model nor estimates a new confidence interval. Calibration curves, ROC-AUC and reporting-only diagnostic fits cannot be reconstructed from these grouped sums alone.

The companion `aggregate-reconstruction/reconstructed-scores.csv` records the executed reconstruction. Original core selector and K20 map Brier values round to 0.235585 and 0.246114. `core-coverage.csv` and `core-phase-counts.csv` are directly readable aggregate tables; `coverage.tex` and `phase-counts.tex` are their article presentations.

## Original figures and complete original comparison tables

The carried-forward original archive provides its pinned `figure-reconstruction/` environment and aggregate inputs. From that directory:

```sh
docker compose run --build --rm render
```

This reconstructs the original aggregate CSVs and figure sets. It does not fit forecasting models, reread source outcomes or independently reproduce scores. The package README names the carried-forward original archive and the separately added new companion; do not substitute a current remote dataset for the pinned source snapshot.

## Documents and access

The editable TeX, BibTeX, adjacent input tables and figure assets reproduce the article PDF using the supplied class/style and the package's build instructions. The Markdown supplement uses the bundled graphics. The data card explains populations, missingness, availability and redistribution scope. No reconstruction command here accesses protected confirmation, runs a repository test suite or establishes a prospective prediction timestamp.

## Completed sensitivity aggregates

`final-sensitivities/all-88-contrasts.csv` contains every declared new contrast, pointwise interval, Monte Carlo quantities and separate-family adjusted decision. `procedure-metrics.csv` separately labels event-equal and row-weighted losses. Its p >0.5 accuracy is a diagnostic, not the original residual decision thresholds. `core-ties-by-block.csv`, `calibration-candidates.csv`, `schedule-gaps.csv` and `removal-parent-metrics.csv` expose the bounded saved-output audits. These summaries permit inspection and figure reconstruction, not fresh bootstrap inference or independent row-level refitting. `specification.json` binds the figure inputs to the saved-report hashes.

The report-to-document renderer is `software/audit-renderers/integrate_final_round.py` in the versioned companion. It reads already-scored private report JSONs, verifies result/completion hashes, emits aggregate CSV/Markdown and draws the figure with Matplotlib; it does not read row-level forecasts or fit models. Re-execution requires the private reports at their recorded local layout or a lawful equivalent supplied with matching identities. The distributed aggregate inputs and figure are available without those private files. The original v4.1 hosted snapshot remains a separate historical release; this dated companion contains the new aggregate outputs listed below.

To reconstruct the new figure from shared aggregates alone (Python 3 with Matplotlib):

```sh
python software/aggregate-renderers/plot_final_sensitivities.py --input final-sensitivities/all-88-contrasts.csv --output reconstructed-final-figure
```

This plots recorded estimates and intervals; it neither recalculates inference nor accesses private reports or source rows.

## Reader-facing saved-output additions

`reader-evidence/later-exact-ties.csv` gives all six original later-cohort procedures with actual tied-subset class counts and original/half-credit row/event decision summaries. It does not expose source IDs and changes no probabilities or loss findings. `online-rating-display.csv` supplies all seven procedures in the expanded online probability-loss figure. Reconstruct that figure using `python software/aggregate-renderers/plot_rating_display.py --input reader-evidence/online-rating-display.csv --output reconstructed-rating-figure` with the supplied plotting environment. The saved-output provenance file binds the aggregate sources and diagnostic implementation by hash. No new inference or model fitting is performed.

The optional `software/audit-renderers/presentation_saved_outputs.py` audit renderer requires explicit `--later`, `--aggregates` and `--output` paths. `--later` refers to restricted frozen forecasts/targets, so this optional command is not runnable from shared aggregates alone. Reviewer score/figure reconstruction above does not need it. No author-local path is embedded in this renderer.
