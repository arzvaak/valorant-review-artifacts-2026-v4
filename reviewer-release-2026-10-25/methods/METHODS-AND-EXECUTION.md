# Methods and execution record

This companion supports inspection of the supplied completed aggregate evidence. It does not contain row-level outcomes, row-level predictions, fitted model binaries, or unrestricted source records.

## Tasks, features, and temporal evaluation

The benchmark has an after-picks map-winner task and a before-play series-winner task. Canonical team orientation is fixed before targets are attached. Series roster and preferred-agent inputs are historical proxies from earlier appearances; actual future lineups and picks are not predictors. The original study uses twelve rolling whole-event windows, keeps Match IDs intact, and fits preprocessing, category vocabularies, model selection, calibration, and mixture weights only on their designated earlier phases. Evaluation blocks are score-only. Original feature layouts, dictionaries, splits, and cutoff logic remain in the carried-forward root-level `schemas/`, `software/`, and `figure-reconstruction/` materials.

The new completed exploratory sensitivity families comprise 20 advancing-history contrasts, 20 changes in the BN–B increment relative to a cloud-fitted frozen-history control, 40 within-policy stricter-clock contrasts, and 8 fixed-weight input-refresh contrasts. They retain the originally selected recipes rather than rerunning the full search. The 6-hour/30-minute and 24-hour/60-minute comparisons jointly vary delay and lead and do not identify either effect separately. Six unavailable earlier feedback outcomes were excluded from history updates after eligibility; no target was imputed and the scoring population rule was unchanged.

Primary losses are event-equal Brier and log loss. Each contrast uses 131,072 paired complete-event resamples and a percentile 95% interval. The centered unstudentized two-sided tail is accompanied by a Clopper–Pearson upper bound on Monte Carlo error; Holm correction is applied separately to the declared 8-, 20-, 20-, and 40-contrast families. Intervals condition on saved fits and do not include full-pipeline retraining/selection uncertainty or fully resolve repeated-team dependence. Non-rejection is not evidence of equivalence.

## Recorded runtime differences

The retained frozen-history controls and fixed-weight/audit outputs were executed on a Tesla T4 with Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1, CUDA 13.0, and 16-mixed neural precision. Advancing-history unfinished units and the stricter-clock arms were continued on an NVIDIA GeForce RTX 4080 SUPER with Python 3.12.14, PyTorch 2.14.0+cu130, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, CUDA 13.0, and 16-mixed neural precision. Both environments recorded XGBoost 3.4.1, CatBoost 1.2.10, DuckDB 1.5.0, Lightning 2.6.6, TorchMetrics 1.9.0, TabM 0.0.3, rtdl-revisiting-models 0.0.2, rtdl-num-embeddings 0.0.12, and CuPy CUDA 13.x 14.2.0.

The completed record therefore mixes hardware and library versions across some saved fits. Completed cloud units retain their cloud identities; unfinished units were continued locally under a separately recorded identity. The published contrasts preserve this limitation and are not described as hardware-matched replication. The original core environment and its aggregate-only reconstruction environment remain separately documented in the carried-forward v4.1 release.

## Interpretation boundary

The event-loss CSVs and other tables are aggregate outputs, not row-level data. The reconstruction commands recover the reported aggregate scores or figures from these supplied values; they do not rescore forecasts, rerun bootstrap inference, retrain models, or establish real-time data availability. Full refitting requires source inputs that are not distributed here and a lawful access arrangement.
