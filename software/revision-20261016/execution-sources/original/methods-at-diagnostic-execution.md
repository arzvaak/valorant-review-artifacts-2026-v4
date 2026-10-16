# Revision methods — internal execution specification v1

Owner authorized execution 16 October 2026. This is internal provenance; manuscript prose describes the science directly.

## Saved-forecast diagnostics

- Artifact roots: configure the external parent artifact tree with `VALORANT_ARTIFACT_ROOT`; the local path is intentionally omitted. Completed historical releases remain read-only.
- Core parent: strong-predictors-20261010-v6. Online-rich parent: online-rich-predictor-20261010-v4. Dynamic parent: dynamic-predictor-20261010-v1. Matched online Elo: dynamic-matched-baselines-20261010-v1. Later parent: later-evaluation-20261011-v1.
- Shared valid original keys/labels only. Broad nonglobal and reviewed-2025 scopes are explicit predicates copied from the executed reports; no outcome-based subset choice.
- Probability clipping [1e-7,1-1e-7]. Equal-event weighting and row weighting reported separately. Exact ties are p=0.5; tie-averaged accuracy gives them 0.5 credit.
- New paired probability-loss intervals: 131072 complete-event resamples, seed 20261016; 512 draws per GPU chunk; event and row-weighted ratio estimators share event draws. Online-rich versus earlier residual, dynamic strength and matched online Elo, for both horizons and broad/reviewed-2025 populations. Core K20 versus matched online Elo and core selector versus matched online Elo additionally quantify matched procedure differences.
- Canonical and doubled/swapped calibration diagnostics for core selector, K20 and prior, and later rich residual: reporting only, no saved-probability modifications. Bin ECE at 5/10/20 bins. Diagnostic parameter/ECE intervals use 2048 paired complete-event resamples, seed 20261017; record convergence and invalid-draw counts. Fit diagnostic coefficients in CUDA float64 using batched two-parameter Newton updates with recorded gradient and conditioning guards; no forecasting model refits.
- Each event resample includes each canonical/swapped pair together at half row weight. Effective event count is unchanged. Mirrored conditional calibration is distinct from canonical conditional calibration and cannot erase canonical reliability concerns.
- Failure accounting includes non-identifiable constants, one-class resamples and nonconvergent fits. Intervals do not conceal excluded draws; unresolved intervals remain unresolved when failure fraction is material (>5%).
- Original primary Monte Carlo/Holm decisions remain unchanged. New uncertainty summaries are exploratory pointwise intervals unless a separately recorded full contrast-family decision is requested; no selective primary-family additions.

## Further method specifications

Rating, matched-input representation, corrected projection and feedback implementations require their exact input/parameter/interface definitions here before launch; their absence does not count as completed work. New source feedback needs an explicit source-bound policy because the existing later contexts are score-only. E12 and locked material remain excluded throughout.
