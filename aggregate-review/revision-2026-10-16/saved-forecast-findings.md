# Saved-forecast revision findings

Original forecasts remain unchanged. New intervals condition on saved fitted procedures and use complete-event resampling; they do not include refitting/selection uncertainty. Loss comparisons below are pointwise exploratory intervals.

## Orientation and exact ties

| Task / procedure | Rows | Exact ties | Ordinary accuracy | Tie-averaged accuracy | Event accuracy | Tie-averaged event accuracy |
|---|---:|---:|---:|---:|---:|---:|
| map_winner / selector | 55943 | 8795 | 58.041% | 59.472% | 59.122% | 59.932% |
| map_winner / block_elo | 55943 | 16421 | 54.048% | 55.822% | 53.591% | 55.152% |
| map_winner / prior | 55943 | 0 | 56.449% | 56.449% | 55.840% | 55.840% |
| map_winner / rich | 55943 | 225 | 61.993% | 62.021% | 61.023% | 61.034% |
| map_winner / online_elo | 55943 | 2386 | 61.404% | 61.598% | 59.665% | 59.973% |
| series_winner / selector | 25186 | 9305 | 54.348% | 57.054% | 53.550% | 55.714% |
| series_winner / block_elo | 25186 | 7705 | 55.158% | 57.482% | 54.601% | 56.454% |
| series_winner / prior | 25186 | 0 | 58.318% | 58.318% | 57.195% | 57.195% |
| series_winner / rich | 25186 | 676 | 63.579% | 63.750% | 61.548% | 61.411% |
| series_winner / online_elo | 25186 | 1263 | 63.837% | 64.093% | 61.596% | 61.835% |

## Matched probability-loss comparisons

Positive differences favor the reference. Online Elo in the original extension is uncalibrated with six-hour feedback; core Elo is separately scalar-calibrated with training-only two-hour history. Their direct difference is a procedure difference, not an isolated refresh effect.

| Task / scope / candidate − reference | Event Brier difference [95% interval] | Event log-loss difference [95% interval] |
|---|---:|---:|
| map_winner / broad_nonglobal / rich − earlier | -0.002019 [-0.002952, -0.001073] | -0.004652 [-0.006724, -0.002550] |
| map_winner / broad_nonglobal / rich − dynamic | -0.008391 [-0.010533, -0.006254] | -0.017642 [-0.022518, -0.012771] |
| map_winner / broad_nonglobal / rich − online_elo | -0.006318 [-0.008743, -0.003858] | -0.013001 [-0.018474, -0.007471] |
| map_winner / broad_nonglobal / block_elo − online_elo | +0.012693 [+0.010669, +0.014712] | +0.027951 [+0.023391, +0.032535] |
| map_winner / broad_nonglobal / selector − online_elo | +0.002164 [-0.000777, +0.005129] | +0.006736 [+0.000075, +0.013441] |
| map_winner / reviewed2025 / rich − earlier | -0.002434 [-0.003577, -0.001441] | -0.005250 [-0.007615, -0.003178] |
| map_winner / reviewed2025 / rich − dynamic | -0.005743 [-0.010150, -0.001415] | -0.009707 [-0.019845, +0.000765] |
| map_winner / reviewed2025 / rich − online_elo | -0.004228 [-0.008515, -0.000111] | -0.006435 [-0.016226, +0.003425] |
| map_winner / reviewed2025 / block_elo − online_elo | +0.015481 [+0.010545, +0.020479] | +0.032486 [+0.021772, +0.043276] |
| map_winner / reviewed2025 / selector − online_elo | +0.000777 [-0.004174, +0.005723] | +0.002209 [-0.008386, +0.012772] |
| series_winner / broad_nonglobal / rich − earlier | -0.003626 [-0.004673, -0.002623] | -0.007713 [-0.009937, -0.005574] |
| series_winner / broad_nonglobal / rich − dynamic | -0.002149 [-0.004255, -0.000029] | -0.004320 [-0.009105, +0.000563] |
| series_winner / broad_nonglobal / rich − online_elo | -0.003671 [-0.006308, -0.001029] | -0.007552 [-0.013480, -0.001585] |
| series_winner / broad_nonglobal / block_elo − online_elo | +0.013341 [+0.010855, +0.015911] | +0.031942 [+0.025703, +0.038564] |
| series_winner / broad_nonglobal / selector − online_elo | +0.013625 [+0.011055, +0.016195] | +0.028927 [+0.023333, +0.034493] |
| series_winner / reviewed2025 / rich − earlier | -0.003091 [-0.003887, -0.002330] | -0.006185 [-0.007779, -0.004664] |
| series_winner / reviewed2025 / rich − dynamic | +0.018841 [+0.012496, +0.025970] | +0.046516 [+0.029248, +0.067705] |
| series_winner / reviewed2025 / rich − online_elo | +0.016159 [+0.009471, +0.023621] | +0.041132 [+0.022946, +0.063252] |
| series_winner / reviewed2025 / block_elo − online_elo | +0.015607 [+0.009335, +0.021774] | +0.033069 [+0.019456, +0.046469] |
| series_winner / reviewed2025 / selector − online_elo | +0.017528 [+0.010839, +0.024162] | +0.036918 [+0.022428, +0.051230] |

## Calibration diagnostic uncertainty

Canonical diagnostics answer calibration on the actually ordered cohort. Mirrored diagnostics retain each pair in the same event resample; symmetry sets their intercept to zero without establishing canonical calibration. Coefficients are reporting only and never alter predictions.

| Task / procedure / orientation | Intercept [95% interval] | Slope [95% interval] | Failed draws / 2048 |
|---|---:|---:|---:|
| map_winner / selector / canonical | 0.1092 [0.0620, 0.1580] | 0.6632 [0.6036, 0.7279] | 0 |
| map_winner / selector / mirrored | 0.0000 [-0.0000, 0.0000] | 0.7027 [0.6445, 0.7646] | 0 |
| map_winner / block_elo / canonical | 0.1694 [0.1220, 0.2207] | 0.5131 [0.4154, 0.6161] | 0 |
| map_winner / block_elo / mirrored | 0.0000 [-0.0000, 0.0000] | 0.6183 [0.5245, 0.7216] | 0 |
| map_winner / prior / canonical | 0.3917 [0.1302, 0.6430] | -0.5700 [-1.4719, 0.3682] | 0 |
| map_winner / prior / mirrored | -0.0000 [-0.0000, 0.0000] | 0.7996 [0.6353, 0.9718] | 0 |
| series_winner / selector / canonical | 0.2099 [0.1466, 0.2699] | 0.5415 [0.4333, 0.6575] | 0 |
| series_winner / selector / mirrored | 0.0000 [-0.0000, 0.0000] | 0.6746 [0.5641, 0.7858] | 0 |
| series_winner / block_elo / canonical | 0.2136 [0.1523, 0.2748] | 0.4657 [0.3697, 0.5622] | 0 |
| series_winner / block_elo / mirrored | 0.0000 [-0.0000, 0.0000] | 0.5624 [0.4681, 0.6600] | 0 |
| series_winner / prior / canonical | 0.4588 [0.1693, 0.7590] | -0.4951 [-1.3509, 0.3142] | 0 |
| series_winner / prior / mirrored | 0.0000 [-0.0000, 0.0000] | 0.7922 [0.6216, 0.9616] | 0 |

## Later retained continuation

| Task | Events | Event-size minimum / median / maximum | Rich Brier (event) | Rich Brier (row) |
|---|---:|---:|---:|---:|
| map_winner | 23 | 8 / 42 / 252 | 0.236592 | 0.227610 |
| series_winner | 23 | 3 / 18 / 111 | 0.245511 | 0.247668 |

The original later series rich–earlier Brier contrast used 131,072 draws and 23 events: approximate centered-tail p=0.037849, Monte Carlo upper bound=0.039527, conservative Holm value=0.474327. The percentile interval and centered-tail test are not inversions of the same construction, and the displayed adjusted decision also includes multiplicity. No specific skewness explanation is established by this arithmetic alone.

## Identity and preservation

Artifact release: revision-20261016-v2 (machine-specific artifact root omitted from this review package). Machine-readable summaries retain per-input identities, actual CUDA memory and elapsed time. All 131,072-draw loss intervals and 2,048-draw calibration intervals preserve event clustering. No predictive fits, threshold changes or source-target regeneration occurred.
