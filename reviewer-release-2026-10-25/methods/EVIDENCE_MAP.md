# Finding-to-evidence map

This map accompanies the manuscript and supplement revision dated 25 October 2026. Original controlled comparisons, the later continuation, and the completed retrospective sensitivities have separate inference families. Probabilities, decisions, fits, and inferential results come from the saved completed analyses. The presentation diagnostics add no hypothesis tests.

| Question / finding | Read first | Inspect the supplied evidence | What this establishes |
|---|---|---|---|
| Numeric history helps selected nonlinear map models | Abstract; controlled results; original information-value figure | Supplement S4 complete BN–B contrasts; original software/aggregate tables | Within-family incremental value under the original frozen-history protocol; not universal model superiority |
| Beating K20 does not establish a contextual advantage | Controlled result table and selector figure | S4 operational contrasts; core event-loss summaries in aggregate-reconstruction | Event-equal Brier/log loss against rating and contextual references, with original correction |
| Historical admission changes the series result | History-admission sensitivity; sensitivity figure | S22; final-sensitivities/all-88-contrasts.csv, families fresh20/change20 | Advancing BN–B effects and change in increments, including logistic-series harm; mixed numerical environments limit policy attribution |
| Two stricter schedule clocks retain the map distinction | Clock sensitivity | S22; clocks40 rows of all-88-contrasts.csv | Within-arm comparisons under jointly changed delay/lead; not measured input-publication eligibility |
| Input refresh helps under unchanged earlier weights | History refresh | S22; refresh8 rows and procedure-metrics.csv | Fixed-estimator input control; the remaining rich-refit contrast also changes reselection/calibration |
| Stronger rating and calibration choices matter | Online table and expanded rating figure | S11/S15; reader-evidence/online-rating-display.csv | Complete seven-procedure absolute probability-loss display; paired intervals remain in S9/S11 |
| Later evidence limits generalization | Later continuation and feedback | S6/S13–S14 complete corrected comparisons | Retrospective later-cohort evidence; neither original nor feedback contrasts establish broad prospective transfer |
| Exact ties cannot be explained by overall class balance | Controlled/later decision qualifications | S22 core audit; S23; reader-evidence/later-exact-ties.csv | Actual tied-subset class counts, fixed-decision/half-credit row/event diagnostics; unchanged probability losses |
| Feature and population availability differ | Data/coverage/phase tables | S2/S7/S19–S20; core-coverage.csv and core-phase-counts.csv | Observed coverage, explicit proxies and chronological decision support; not verified live publication |

## Reviewer reconstruction routes

From the aggregate companion root:

```sh
python software/aggregate-renderers/reconstruct_event_scores.py --input aggregate-reconstruction/event-forecast-losses.csv --output reconstructed-scores.csv
python software/aggregate-renderers/plot_final_sensitivities.py --input final-sensitivities/all-88-contrasts.csv --output reconstructed-history-figure
python software/aggregate-renderers/plot_rating_display.py --input reader-evidence/online-rating-display.csv --output reconstructed-rating-figure
```

The first uses supplied complete-event loss/correct-count summaries to reconstruct primary scores. The plots use already-reported aggregate estimates and intervals; they do not recalculate bootstrap inference or read private source rows. See RECONSTRUCTION.md, METHODS-AND-EXECUTION.md and the supplied environment record for dependencies and the original figure reconstruction route.

## Access boundary

The shared software and aggregates support the score and figure reconstruction routes above and inspection of the comparison families. They are not a row-level training dataset, do not regenerate the original features, and do not support full forecasting refits without compatible, lawfully obtained source inputs. Raw captures, source row keys, per-target probabilities and labels, and model binaries are not included. The data card and feature, split, and environment records describe what additional inputs a refit would require. These retrospective results do not establish prospective prediction availability or grant a license to redistribute source data.
