# Finding-to-evidence map

This map accompanies the matching manuscript and supplement. Original controlled comparisons, later continuation and new retrospective sensitivities have separate inference families. All probabilities, decisions, fits and inferential results are retained from completed execution. New presentation diagnostics add no hypothesis tests.

| Question / finding | Read first | Inspect the supplied evidence | What this establishes |
|---|---|---|---|
| Numeric history helps selected nonlinear map models | Abstract; controlled results; original information-value figure | Supplement S4 complete BN–B contrasts; original software and aggregate tables | Within-family incremental value under the original frozen-history protocol; not universal model superiority |
| Beating K20 does not establish a contextual advantage | Controlled result table and selector figure | S4 operational contrasts; `aggregates/aggregate-reconstruction/event-forecast-losses.csv` | Event-equal Brier/log loss against rating and contextual references, with the original correction |
| Historical admission changes the series result | History-admission sensitivity and sensitivity figure | S22; `aggregates/final-sensitivities/all-88-contrasts.csv`, families `fresh20` and `change20` | Advancing BN–B effects and changes in increments, including logistic-series harm; mixed numerical environments limit policy attribution |
| Two stricter schedule clocks retain the map distinction | Clock sensitivity | S22; `clocks40` rows of `aggregates/final-sensitivities/all-88-contrasts.csv` | Within-arm comparisons under jointly changed delay and lead; not measured input-publication eligibility |
| Input refresh helps under unchanged earlier weights | History refresh | S22; `refresh8` rows and `aggregates/final-sensitivities/procedure-metrics.csv` | Fixed-estimator input control; the remaining rich-refit contrast also changes reselection and calibration |
| Stronger rating and calibration choices matter | Main online figure and complete score table | S26; `aggregates/reader-evidence/online-rating-display.csv`; paired intervals in S9/S11 | Complete seven-procedure absolute probability-loss display; uncertainty is reported separately |
| Later evidence limits generalization | Later continuation and feedback | S6/S13–S14 complete corrected comparisons | Retrospective later-cohort evidence; neither original nor feedback contrasts establish broad prospective transfer |
| Exact ties cannot be explained by overall class balance | Controlled and later decision qualifications | S22 core audit; S23; `aggregates/reader-evidence/later-exact-ties.csv` | Actual tied-subset class counts and fixed-decision/half-credit row/event diagnostics; probability losses are unchanged |
| Feature and population availability differ | Data/coverage/phase tables | S2/S7/S19–S20; `aggregates/aggregate-reconstruction/core-coverage.csv` and `core-phase-counts.csv` | Observed coverage, explicit proxies and chronological decision support; not verified live publication |

## Reviewer reconstruction routes

Run these commands from the `reviewer-release-2026-10-25/` directory:

```sh
python software/aggregate-renderers/reconstruct_event_scores.py \
  --input aggregates/aggregate-reconstruction/event-forecast-losses.csv \
  --output reconstructed-scores.csv

python software/aggregate-renderers/plot_final_sensitivities.py \
  --input aggregates/final-sensitivities/all-88-contrasts.csv \
  --output reconstructed-history-figure

python software/aggregate-renderers/plot_rating_display.py \
  --input aggregates/reader-evidence/online-rating-display.csv \
  --output reconstructed-rating-figure
```

The first command uses supplied complete-event loss and correct-count summaries to reconstruct primary scores. The plots use already-reported aggregate estimates and intervals; they do not recalculate bootstrap inference or read restricted source rows. See `methods/RECONSTRUCTION.md`, `methods/METHODS-AND-EXECUTION.md` and the supplied environment record for dependencies and the original figure reconstruction route.

## Access boundary

The shared software and aggregates support these score and figure reconstruction routes and inspection of the comparison families. They are not a row-level training dataset, do not regenerate the original features, and do not support full forecasting refits without compatible, lawfully obtained source inputs. Raw captures, source row keys, per-target probabilities and labels, and model binaries are not included. The data card and feature, split and environment records describe what additional inputs a refit would require. These retrospective results do not establish prospective prediction availability or grant a license to redistribute source data.
