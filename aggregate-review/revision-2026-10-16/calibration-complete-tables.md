# Complete calibration diagnostic intervals

Reporting-only canonical fits never change probabilities. Pointwise 95% complete-event intervals use 2048 draws. Exact constants have non-identifiable slopes; they remain unresolved.

| Task/procedure | Intercept [95% interval] | Slope [95% interval] | Failed draws |
|---|---:|---:|---:|
| map_winner/selector/canonical | 0.109220 [0.062006, 0.157953] | 0.663177 [0.603641, 0.727937] | 0 / 2048 |
| map_winner/selector/mirrored | 0.000000 [-0.000000, 0.000000] | 0.702686 [0.644483, 0.764619] | 0 / 2048 |
| map_winner/block_elo/canonical | 0.169376 [0.122005, 0.220668] | 0.513110 [0.415393, 0.616096] | 0 / 2048 |
| map_winner/block_elo/mirrored | 0.000000 [-0.000000, 0.000000] | 0.618263 [0.524481, 0.721585] | 0 / 2048 |
| map_winner/prior/canonical | 0.391728 [0.130174, 0.643025] | -0.569969 [-1.471855, 0.368238] | 0 / 2048 |
| map_winner/prior/mirrored | -0.000000 [-0.000000, 0.000000] | 0.799578 [0.635320, 0.971823] | 0 / 2048 |
| series_winner/selector/canonical | 0.209908 [0.146607, 0.269943] | 0.541466 [0.433296, 0.657536] | 0 / 2048 |
| series_winner/selector/mirrored | 0.000000 [-0.000000, 0.000000] | 0.674552 [0.564138, 0.785803] | 0 / 2048 |
| series_winner/block_elo/canonical | 0.213553 [0.152255, 0.274777] | 0.465717 [0.369708, 0.562228] | 0 / 2048 |
| series_winner/block_elo/mirrored | 0.000000 [-0.000000, 0.000000] | 0.562372 [0.468089, 0.660008] | 0 / 2048 |
| series_winner/prior/canonical | 0.458796 [0.169291, 0.758987] | -0.495141 [-1.350934, 0.314211] | 0 / 2048 |
| series_winner/prior/mirrored | 0.000000 [-0.000000, 0.000000] | 0.792164 [0.621593, 0.961572] | 0 / 2048 |
| map_winner/rich/canonical | 0.111012 [0.062691, 0.159380] | 0.815548 [0.756292, 0.875172] | 0 / 2048 |
| map_winner/earlier/canonical | 0.116236 [0.067815, 0.163239] | 0.794681 [0.737327, 0.852012] | 0 / 2048 |
| map_winner/dynamic/canonical | 0.152521 [0.105512, 0.200325] | 1.273510 [1.163905, 1.378125] | 0 / 2048 |
| map_winner/online_elo/canonical | 0.090801 [0.041464, 0.141304] | 0.962392 [0.885253, 1.047697] | 0 / 2048 |
| series_winner/rich/canonical | 0.134692 [0.071043, 0.194135] | 0.954339 [0.852928, 1.055996] | 0 / 2048 |
| series_winner/earlier/canonical | 0.143205 [0.081932, 0.204286] | 0.870016 [0.772381, 0.970367] | 0 / 2048 |
| series_winner/dynamic/canonical | 0.188772 [0.128735, 0.248209] | 1.487498 [1.340360, 1.636275] | 0 / 2048 |
| series_winner/online_elo/canonical | 0.125748 [0.065835, 0.188882] | 1.234079 [1.088631, 1.382775] | 0 / 2048 |
| map_winner/later_rich/canonical | 0.125731 [-0.035443, 0.287240] | 0.509548 [0.229674, 0.788548] | 0 / 2048 |
| map_winner/later_rich/mirrored | 0.000000 [-0.000000, 0.000000] | 0.515058 [0.244408, 0.795337] | 0 / 2048 |
| series_winner/later_rich/canonical | 0.150673 [-0.151981, 0.494973] | 0.709937 [0.040671, 1.684707] | 0 / 2048 |
| series_winner/later_rich/mirrored | 0.000000 [-0.000000, 0.000000] | 0.857529 [0.298747, 1.962713] | 0 / 2048 |

## ECE bin sensitivity

| Task/procedure | Bins | ECE [95% interval] |
|---|---:|---:|
| map_winner/selector/canonical | 5 | 0.038070 [0.027022, 0.048156] |
| map_winner/selector/canonical | 10 | 0.038070 [0.029435, 0.048807] |
| map_winner/selector/canonical | 20 | 0.038070 [0.033320, 0.051181] |
| map_winner/selector/mirrored | 5 | 0.021333 [0.014399, 0.028108] |
| map_winner/selector/mirrored | 10 | 0.023597 [0.017737, 0.032680] |
| map_winner/selector/mirrored | 20 | 0.025182 [0.019745, 0.035444] |
| map_winner/block_elo/canonical | 5 | 0.046876 [0.035964, 0.057721] |
| map_winner/block_elo/canonical | 10 | 0.046876 [0.036230, 0.057721] |
| map_winner/block_elo/canonical | 20 | 0.046893 [0.037602, 0.058472] |
| map_winner/block_elo/mirrored | 5 | 0.014164 [0.009701, 0.018761] |
| map_winner/block_elo/mirrored | 10 | 0.019521 [0.012223, 0.028117] |
| map_winner/block_elo/mirrored | 20 | 0.021601 [0.014557, 0.030120] |
| map_winner/prior/canonical | 5 | 0.009973 [0.000802, 0.021117] |
| map_winner/prior/canonical | 10 | 0.009973 [0.000802, 0.021117] |
| map_winner/prior/canonical | 20 | 0.009973 [0.000802, 0.021117] |
| map_winner/prior/mirrored | 5 | 0.000000 [0.000000, 0.000000] |
| map_winner/prior/mirrored | 10 | 0.009973 [0.000802, 0.021117] |
| map_winner/prior/mirrored | 20 | 0.009973 [0.000802, 0.021117] |
| series_winner/selector/canonical | 5 | 0.056552 [0.042634, 0.071211] |
| series_winner/selector/canonical | 10 | 0.056557 [0.042931, 0.071329] |
| series_winner/selector/canonical | 20 | 0.056643 [0.044999, 0.071838] |
| series_winner/selector/mirrored | 5 | 0.014795 [0.008167, 0.021396] |
| series_winner/selector/mirrored | 10 | 0.024312 [0.014306, 0.034929] |
| series_winner/selector/mirrored | 20 | 0.027610 [0.020284, 0.038355] |
| series_winner/block_elo/canonical | 5 | 0.055098 [0.041333, 0.069401] |
| series_winner/block_elo/canonical | 10 | 0.055873 [0.043834, 0.071164] |
| series_winner/block_elo/canonical | 20 | 0.059189 [0.046620, 0.073638] |
| series_winner/block_elo/mirrored | 5 | 0.018130 [0.011923, 0.024800] |
| series_winner/block_elo/mirrored | 10 | 0.025092 [0.016381, 0.036261] |
| series_winner/block_elo/mirrored | 20 | 0.025794 [0.019643, 0.038466] |
| series_winner/prior/canonical | 5 | 0.012406 [0.007376, 0.028359] |
| series_winner/prior/canonical | 10 | 0.012406 [0.007376, 0.028359] |
| series_winner/prior/canonical | 20 | 0.012406 [0.007376, 0.028359] |
| series_winner/prior/mirrored | 5 | 0.010876 [0.004545, 0.017702] |
| series_winner/prior/mirrored | 10 | 0.012406 [0.007376, 0.028359] |
| series_winner/prior/mirrored | 20 | 0.012406 [0.007376, 0.028359] |
| map_winner/rich/canonical | 5 | 0.024875 [0.015439, 0.035433] |
| map_winner/rich/canonical | 10 | 0.032415 [0.023710, 0.043103] |
| map_winner/rich/canonical | 20 | 0.032415 [0.025777, 0.044037] |
| map_winner/earlier/canonical | 5 | 0.026651 [0.018325, 0.038376] |
| map_winner/earlier/canonical | 10 | 0.035496 [0.027352, 0.046392] |
| map_winner/earlier/canonical | 20 | 0.035969 [0.029115, 0.047632] |
| map_winner/dynamic/canonical | 5 | 0.042314 [0.032061, 0.053561] |
| map_winner/dynamic/canonical | 10 | 0.042328 [0.033464, 0.054190] |
| map_winner/dynamic/canonical | 20 | 0.042438 [0.035252, 0.055005] |
| map_winner/online_elo/canonical | 5 | 0.020403 [0.010601, 0.032324] |
| map_winner/online_elo/canonical | 10 | 0.021355 [0.013730, 0.033424] |
| map_winner/online_elo/canonical | 20 | 0.023343 [0.017612, 0.036930] |
| series_winner/rich/canonical | 5 | 0.029453 [0.017670, 0.043070] |
| series_winner/rich/canonical | 10 | 0.032080 [0.023088, 0.047853] |
| series_winner/rich/canonical | 20 | 0.037120 [0.029272, 0.053132] |
| series_winner/earlier/canonical | 5 | 0.028854 [0.018748, 0.043739] |
| series_winner/earlier/canonical | 10 | 0.031163 [0.021208, 0.046283] |
| series_winner/earlier/canonical | 20 | 0.037737 [0.030234, 0.054624] |
| series_winner/dynamic/canonical | 5 | 0.058110 [0.044552, 0.071959] |
| series_winner/dynamic/canonical | 10 | 0.058110 [0.044580, 0.071960] |
| series_winner/dynamic/canonical | 20 | 0.058201 [0.046806, 0.072581] |
| series_winner/online_elo/canonical | 5 | 0.041048 [0.027312, 0.054801] |
| series_winner/online_elo/canonical | 10 | 0.042629 [0.033535, 0.057401] |
| series_winner/online_elo/canonical | 20 | 0.043088 [0.035718, 0.059400] |
| map_winner/later_rich/canonical | 5 | 0.090057 [0.043488, 0.158773] |
| map_winner/later_rich/canonical | 10 | 0.090057 [0.049911, 0.162813] |
| map_winner/later_rich/canonical | 20 | 0.093132 [0.070663, 0.170367] |
| map_winner/later_rich/mirrored | 5 | 0.073784 [0.026072, 0.136656] |
| map_winner/later_rich/mirrored | 10 | 0.085003 [0.041612, 0.145957] |
| map_winner/later_rich/mirrored | 20 | 0.085181 [0.056283, 0.149671] |
| series_winner/later_rich/canonical | 5 | 0.052228 [0.015866, 0.123467] |
| series_winner/later_rich/canonical | 10 | 0.072200 [0.034209, 0.141345] |
| series_winner/later_rich/canonical | 20 | 0.080927 [0.045148, 0.147811] |
| series_winner/later_rich/mirrored | 5 | 0.005341 [0.001258, 0.024218] |
| series_winner/later_rich/mirrored | 10 | 0.071225 [0.029106, 0.134802] |
| series_winner/later_rich/mirrored | 20 | 0.080339 [0.037560, 0.139479] |
