# Stronger rating references and matched refresh comparison

All rows use the original valid broad nonglobal evaluation targets. Probabilities are raw or separately calibrated on the original calibration partition. No evaluation-outcome tuning. Player Gaussian is the declared diagonal TrueSkill-model adaptation; incomplete prior series rosters fall back to team Glicko-2.

| Task / regime / rating | Event Brier | Event log loss | Ordinary accuracy | Event accuracy |
|---|---:|---:|---:|---:|
| map_winner / online_raw / elo_k20 | 0.233421 | 0.658643 | 61.40% | 59.66% |
| map_winner / online_raw / glicko2 | 0.240854 | 0.687036 | 62.24% | 60.26% |
| map_winner / online_raw / player_gaussian | 0.242518 | 0.707447 | 62.93% | 61.63% |
| map_winner / online_calibrated / elo_k20 | 0.234683 | 0.662111 | 61.40% | 59.66% |
| map_winner / online_calibrated / glicko2 | 0.234258 | 0.662775 | 62.24% | 60.26% |
| map_winner / online_calibrated / player_gaussian | 0.231597 | 0.658968 | 62.93% | 61.63% |
| map_winner / frozen_raw / elo_k20 | 0.245530 | 0.685121 | 54.08% | 53.63% |
| map_winner / frozen_raw / glicko2 | 0.264662 | 0.751307 | 54.38% | 53.86% |
| map_winner / frozen_raw / player_gaussian | 0.261427 | 0.778903 | 58.80% | 58.33% |
| map_winner / frozen_calibrated / elo_k20 | 0.246127 | 0.686656 | 54.08% | 53.63% |
| map_winner / frozen_calibrated / glicko2 | 0.247812 | 0.692813 | 54.38% | 53.86% |
| map_winner / frozen_calibrated / player_gaussian | 0.241002 | 0.681587 | 58.80% | 58.33% |
| series_winner / online_raw / elo_k20 | 0.230961 | 0.653703 | 63.84% | 61.60% |
| series_winner / online_raw / glicko2 | 0.226460 | 0.650644 | 65.25% | 63.06% |
| series_winner / online_raw / player_gaussian | 0.227352 | 0.656753 | 64.80% | 62.99% |
| series_winner / online_calibrated / elo_k20 | 0.232486 | 0.660627 | 63.84% | 61.60% |
| series_winner / online_calibrated / glicko2 | 0.224916 | 0.642348 | 65.25% | 63.06% |
| series_winner / online_calibrated / player_gaussian | 0.226002 | 0.646070 | 64.80% | 62.99% |
| series_winner / frozen_raw / elo_k20 | 0.242628 | 0.678340 | 54.72% | 53.98% |
| series_winner / frozen_raw / glicko2 | 0.255975 | 0.728066 | 55.55% | 54.92% |
| series_winner / frozen_raw / player_gaussian | 0.259622 | 0.749201 | 55.14% | 54.45% |
| series_winner / frozen_calibrated / elo_k20 | 0.245069 | 0.687773 | 54.72% | 53.98% |
| series_winner / frozen_calibrated / glicko2 | 0.245607 | 0.689469 | 55.55% | 54.92% |
| series_winner / frozen_calibrated / player_gaussian | 0.246162 | 0.691908 | 55.14% | 54.45% |

## Paired differences

Intervals are pointwise, 131,072 complete-event draws, seed 20261018. Negative loss differences favor the first procedure.

| Task / mode / candidate − reference | Brier difference [95% interval] | Log-loss difference [95% interval] |
|---|---:|---:|
| map_winner / raw / online_elo_k20 − frozen_elo_k20 | -0.012109 [-0.013902, -0.010321] | -0.026479 [-0.030430, -0.022555] |
| map_winner / raw / online_glicko2 − frozen_glicko2 | -0.023807 [-0.028281, -0.019413] | -0.064271 [-0.076316, -0.052391] |
| map_winner / raw / online_player_gaussian − frozen_player_gaussian | -0.018908 [-0.023619, -0.014203] | -0.071456 [-0.086437, -0.056607] |
| map_winner / raw / online_glicko2 − online_elo_k20 | +0.007433 [+0.004214, +0.010727] | +0.028394 [+0.019848, +0.037218] |
| map_winner / raw / online_player_gaussian − online_elo_k20 | +0.009097 [+0.004282, +0.014026] | +0.048805 [+0.034506, +0.063966] |
| map_winner / calibrated / online_elo_k20 − frozen_elo_k20 | -0.011444 [-0.013385, -0.009490] | -0.024545 [-0.028993, -0.020077] |
| map_winner / calibrated / online_glicko2 − frozen_glicko2 | -0.013553 [-0.016451, -0.010739] | -0.030038 [-0.036841, -0.023488] |
| map_winner / calibrated / online_player_gaussian − frozen_player_gaussian | -0.009405 [-0.012428, -0.006422] | -0.022618 [-0.030667, -0.014472] |
| map_winner / calibrated / online_glicko2 − online_elo_k20 | -0.000424 [-0.002370, +0.001545] | +0.000664 [-0.004002, +0.005396] |
| map_winner / calibrated / online_player_gaussian − online_elo_k20 | -0.003086 [-0.006329, +0.000267] | -0.003143 [-0.011209, +0.005490] |
| series_winner / raw / online_elo_k20 − frozen_elo_k20 | -0.011667 [-0.013378, -0.009963] | -0.024637 [-0.028290, -0.021009] |
| series_winner / raw / online_glicko2 − frozen_glicko2 | -0.029515 [-0.034912, -0.024310] | -0.077422 [-0.092112, -0.063471] |
| series_winner / raw / online_player_gaussian − frozen_player_gaussian | -0.032270 [-0.038364, -0.026348] | -0.092448 [-0.109798, -0.075721] |
| series_winner / raw / online_glicko2 − online_elo_k20 | -0.004501 [-0.008637, -0.000336] | -0.003059 [-0.013800, +0.007825] |
| series_winner / raw / online_player_gaussian − online_elo_k20 | -0.003610 [-0.008133, +0.000936] | +0.003050 [-0.008795, +0.015041] |
| series_winner / calibrated / online_elo_k20 − frozen_elo_k20 | -0.012583 [-0.015444, -0.009816] | -0.027146 [-0.034930, -0.019988] |
| series_winner / calibrated / online_glicko2 − frozen_glicko2 | -0.020691 [-0.024500, -0.016920] | -0.047121 [-0.055962, -0.038350] |
| series_winner / calibrated / online_player_gaussian − frozen_player_gaussian | -0.020160 [-0.024033, -0.016397] | -0.045837 [-0.055495, -0.036502] |
| series_winner / calibrated / online_glicko2 − online_elo_k20 | -0.007570 [-0.010408, -0.004703] | -0.018279 [-0.025312, -0.011130] |
| series_winner / calibrated / online_player_gaussian − online_elo_k20 | -0.006484 [-0.009970, -0.002988] | -0.014557 [-0.023256, -0.005808] |

Bound outputs: [`rating-comparison/results.json`](../../figure-reconstruction/package/data/revision/rating-comparison/results.json). Online identity: `{'code': 'fccfc2d3d138eeb3d9c25e1996c9f20cd2880a298afc3b7895671bc8eb4f75ac', 'methods': 'df4cdd80ef5c6ab64830bd679b5d01d882ebb4a931f05e33baa356de67b23006', 'regime': 'online', 'torch': '2.14.0+cu130'}`. Frozen identity: `{'code': '1242aa0f4afcf1ae963ec8ca0ed76d3801f20f22697649b1dffd2fb33ca1d933', 'methods': '6dbb4906ca21f9f4716154145d2c6727cf998d85aba0975401be4d96de0693a2', 'regime': 'frozen', 'delay_seconds': 21600, 'torch': '2.14.0+cu130'}`.
