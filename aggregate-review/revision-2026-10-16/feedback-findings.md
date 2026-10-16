# Matched retained-feedback continuation

18 explicitly regional source-header events admit feedback; five unresolved events remain score-only. All 23 events remain scored. Weights, preprocessing, calibration and decisions retain their original packages. All intervals are conditional complete-event intervals with 131072 draws, seed 20261021. The new 16-contrast family uses its own Monte Carlo bound/Holm correction; no corrected decisions pass.

## Common-key scores

| Task/arm/policy | Rows/events | Event Brier | Event log loss | Row accuracy | Event accuracy |
|---|---:|---:|---:|---:|---:|
| map_winner/rich_residual/feedback | 1613/23 | 0.223034 | 0.634264 | 0.639182 | 0.591125 |
| map_winner/rich_residual/score_only | 1613/23 | 0.236592 | 0.678430 | 0.626782 | 0.597009 |
| map_winner/original_residual/feedback | 1613/23 | 0.240217 | 0.685030 | 0.639802 | 0.592609 |
| map_winner/original_residual/score_only | 1613/23 | 0.242879 | 0.693062 | 0.626782 | 0.583819 |
| map_winner/dynamic_strength/feedback | 1613/23 | 0.241656 | 0.675859 | 0.580905 | 0.557284 |
| map_winner/dynamic_strength/score_only | 1613/23 | 0.247051 | 0.687163 | 0.506510 | 0.527458 |
| map_winner/online_elo_k20/feedback | 1613/23 | 0.242124 | 0.676820 | 0.580285 | 0.544546 |
| map_winner/online_elo_k20/score_only | 1613/23 | 0.246596 | 0.686134 | 0.513949 | 0.513510 |
| series_winner/rich_residual/feedback | 734/23 | 0.235636 | 0.663111 | 0.557221 | 0.507153 |
| series_winner/rich_residual/score_only | 734/23 | 0.245511 | 0.684155 | 0.498638 | 0.475220 |
| series_winner/original_residual/feedback | 734/23 | 0.243620 | 0.679835 | 0.506812 | 0.460096 |
| series_winner/original_residual/score_only | 734/23 | 0.248806 | 0.690887 | 0.491826 | 0.454089 |
| series_winner/dynamic_strength/feedback | 734/23 | 0.237716 | 0.667542 | 0.566757 | 0.538503 |
| series_winner/dynamic_strength/score_only | 734/23 | 0.245980 | 0.684959 | 0.479564 | 0.501239 |
| series_winner/online_elo_k20/feedback | 734/23 | 0.242848 | 0.678446 | 0.574932 | 0.526953 |
| series_winner/online_elo_k20/score_only | 734/23 | 0.247347 | 0.687624 | 0.497275 | 0.499430 |

## All paired feedback-minus-score-only contrasts

| Contrast | Event difference | Pointwise 95% interval | Row difference | Row 95% interval | Conservative Holm |
|---|---:|---|---:|---|---:|
| map_winner/rich_residual/brier | -0.013558 | [-0.033160, -0.000245] | -0.015797 | [-0.023439, -0.005479] | 0.314147 |
| map_winner/rich_residual/log_loss | -0.044166 | [-0.104286, -0.004808] | -0.045100 | [-0.064565, -0.019413] | 0.314147 |
| map_winner/original_residual/brier | -0.002661 | [-0.006147, +0.000455] | -0.006733 | [-0.010599, -0.002357] | 0.314147 |
| map_winner/original_residual/log_loss | -0.008032 | [-0.016399, -0.000405] | -0.016128 | [-0.024929, -0.006591] | 0.314147 |
| map_winner/dynamic_strength/brier | -0.005395 | [-0.010825, -0.000548] | -0.012095 | [-0.017991, -0.004165] | 0.314147 |
| map_winner/dynamic_strength/log_loss | -0.011303 | [-0.022593, -0.001272] | -0.025264 | [-0.037695, -0.008610] | 0.314147 |
| map_winner/online_elo_k20/brier | -0.004472 | [-0.009026, -0.000391] | -0.010052 | [-0.014998, -0.003531] | 0.314147 |
| map_winner/online_elo_k20/log_loss | -0.009315 | [-0.018675, -0.000939] | -0.020882 | [-0.031214, -0.007307] | 0.314147 |
| series_winner/rich_residual/brier | -0.009874 | [-0.018371, -0.002052] | -0.019259 | [-0.029165, -0.006725] | 0.202771 |
| series_winner/rich_residual/log_loss | -0.021043 | [-0.038814, -0.004710] | -0.040976 | [-0.062064, -0.014367] | 0.196969 |
| series_winner/original_residual/brier | -0.005186 | [-0.010075, -0.000734] | -0.010639 | [-0.015878, -0.003911] | 0.281201 |
| series_winner/original_residual/log_loss | -0.011052 | [-0.021145, -0.001840] | -0.022671 | [-0.033800, -0.008378] | 0.264080 |
| series_winner/dynamic_strength/brier | -0.008264 | [-0.014849, -0.002675] | -0.015674 | [-0.024185, -0.005232] | 0.159040 |
| series_winner/dynamic_strength/log_loss | -0.017416 | [-0.031450, -0.005609] | -0.033125 | [-0.051328, -0.010916] | 0.159040 |
| series_winner/online_elo_k20/brier | -0.004498 | [-0.008266, -0.001508] | -0.008465 | [-0.013486, -0.002645] | 0.175819 |
| series_winner/online_elo_k20/log_loss | -0.009178 | [-0.016949, -0.003038] | -0.017337 | [-0.027710, -0.005335] | 0.175819 |

## Accuracy differences

| Task/arm | Row difference [95% interval] | Event difference [95% interval] |
|---|---:|---:|
| map_winner/rich_residual | +0.012399 [-0.017506, +0.034269] | -0.005884 [-0.023454, +0.009814] |
| map_winner/original_residual | +0.013019 [+0.006039, +0.019981] | +0.008790 [+0.003168, +0.015351] |
| map_winner/dynamic_strength | +0.074396 [+0.030504, +0.106406] | +0.029826 [-0.009573, +0.067211] |
| map_winner/online_elo_k20 | +0.066336 [+0.015926, +0.105301] | +0.031036 [-0.002176, +0.065411] |
| series_winner/rich_residual | +0.058583 [+0.021248, +0.087488] | +0.031933 [+0.011732, +0.054584] |
| series_winner/original_residual | +0.014986 [+0.001938, +0.025411] | +0.006007 [-0.000568, +0.013693] |
| series_winner/dynamic_strength | +0.087193 [+0.024601, +0.137150] | +0.037264 [-0.010247, +0.084938] |
| series_winner/online_elo_k20 | +0.077657 [+0.017283, +0.123746] | +0.027523 [-0.009782, +0.066819] |

## Acquired history support

2196 of 2386 all-context predictions have previously released new-cohort feedback. 1478 admitted map overview records provide the available performance fields; this is not a recovery of all economy source sections. Valid-target coverage below differs from the original all-context inventory.

| Task/policy | Valid rows | Both zero team exposure | Both full player-state coverage | Either zero player-state coverage |
|---|---:|---:|---:|---:|
| map_winner/score_only | 1613 | 1115 | 201 | 522 |
| map_winner/feedback | 1613 | 416 | 792 | 248 |
| series_winner/score_only | 734 | 518 | 27 | 707 |
| series_winner/feedback | 734 | 208 | 321 | 413 |
