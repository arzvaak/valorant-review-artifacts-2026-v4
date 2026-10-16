# Controlled player inference specification

Recorded while fitting is in progress, before extension evaluation scoring. No extension evaluation losses have been inspected.

All contrasts use original valid broad nonglobal evaluation keys, with complete blocks and retained fallbacks. Three-seed means are separately scalar-calibrated on the original calibration partition before scoring. Contrast families are independent of the original primary/player/removal families.

- Pool family: mean−sum, attention−sum, attention−mean, sum−summary and attention−summary within each corrected BN/BNC/BNL/BNLC representation: 5 × 4 representations × 2 horizons × 2 losses = 80.
- Channel family: BNC−BN and BNLC−BNL for C; BNL−BN and BNLC−BNC for L, within each of four pools: 4 × 4 × 2 × 2 = 64.
- Projection family: corrected BNLC sum−original BNLC sum, both horizons/losses = 4.

Use 131072 complete-event paired resamples, seed 20261020 plus stable contrast ordinal. Centered-bootstrap approximate tails, simultaneous 99% binomial Monte Carlo upper bounds within each family (budgets 80/64/4), and Holm adjustment within each actual family use the original executed formulas. No row/event omission based on results, alternative recipe selection or primary-family merging. Pointwise intervals condition on completed fits; they do not include refitting uncertainty.

Report all contrast rows and parameter counts, actual epoch/batch telemetry, sparse fallbacks, headshot projection identity, and all failures. Keep gains, harms and unresolved comparisons equally visible.
