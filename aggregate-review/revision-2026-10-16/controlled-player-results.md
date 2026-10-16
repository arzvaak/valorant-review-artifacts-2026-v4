# Controlled player-pooling and channel results

All completed conditions are retained. Sum/mean/attention share the encoder/head dimensions; summary pooling uses the same encoded information but a wider head. C and L channels are independently crossed. Original studies and their decisions remain unchanged.

## Complete family accounting

| Family | Contrasts | Corrected benefits | Corrected harms | Unresolved |
|---|---:|---:|---:|---:|
| pool80 | 80 | 0 | 1 | 79 |
| channel64 | 64 | 0 | 0 | 64 |
| projection4 | 4 | 0 | 0 | 4 |

## Full calibrated metrics

| Task / package | Event Brier | Event log loss | Row accuracy | Event accuracy |
|---|---:|---:|---:|---:|
| map_winner / BN-sum | 0.234149 | 0.661430 | 57.87% | 59.13% |
| map_winner / BN-mean | 0.234662 | 0.662564 | 57.84% | 59.07% |
| map_winner / BN-attention | 0.234730 | 0.662717 | 57.84% | 59.11% |
| map_winner / BN-summary | 0.234363 | 0.661566 | 57.83% | 59.21% |
| map_winner / BNC-sum | 0.234607 | 0.662689 | 58.09% | 58.96% |
| map_winner / BNC-mean | 0.234866 | 0.663388 | 58.07% | 59.15% |
| map_winner / BNC-attention | 0.234799 | 0.663209 | 58.05% | 59.03% |
| map_winner / BNC-summary | 0.234454 | 0.661950 | 58.12% | 59.25% |
| map_winner / BNL-sum | 0.234053 | 0.661226 | 58.16% | 59.04% |
| map_winner / BNL-mean | 0.234506 | 0.662019 | 57.97% | 58.83% |
| map_winner / BNL-attention | 0.234570 | 0.662135 | 57.99% | 58.81% |
| map_winner / BNL-summary | 0.234062 | 0.660744 | 58.08% | 58.94% |
| map_winner / BNLC-sum | 0.234339 | 0.661833 | 58.24% | 59.09% |
| map_winner / BNLC-mean | 0.234373 | 0.661807 | 58.17% | 58.85% |
| map_winner / BNLC-attention | 0.234420 | 0.661908 | 58.10% | 58.73% |
| map_winner / BNLC-summary | 0.234175 | 0.661264 | 58.29% | 59.16% |
| map_winner / BNLC-original_sum | 0.234128 | 0.661135 | 58.21% | 58.88% |
| series_winner / BN-sum | 0.243741 | 0.680670 | 54.37% | 53.66% |
| series_winner / BN-mean | 0.244031 | 0.681307 | 54.23% | 53.55% |
| series_winner / BN-attention | 0.243932 | 0.681100 | 54.31% | 53.66% |
| series_winner / BN-summary | 0.243492 | 0.680114 | 54.56% | 53.77% |
| series_winner / BNC-sum | 0.245220 | 0.683917 | 54.01% | 53.34% |
| series_winner / BNC-mean | 0.245002 | 0.683424 | 54.13% | 53.34% |
| series_winner / BNC-attention | 0.245189 | 0.683835 | 54.03% | 53.17% |
| series_winner / BNC-summary | 0.244764 | 0.682875 | 54.24% | 53.46% |
| series_winner / BNL-sum | 0.244423 | 0.682168 | 54.36% | 53.81% |
| series_winner / BNL-mean | 0.243983 | 0.681233 | 54.24% | 53.69% |
| series_winner / BNL-attention | 0.243985 | 0.681248 | 54.31% | 53.76% |
| series_winner / BNL-summary | 0.244253 | 0.682499 | 54.57% | 53.91% |
| series_winner / BNLC-sum | 0.244672 | 0.682644 | 54.30% | 53.36% |
| series_winner / BNLC-mean | 0.244789 | 0.682837 | 54.17% | 53.32% |
| series_winner / BNLC-attention | 0.244466 | 0.682162 | 54.32% | 53.56% |
| series_winner / BNLC-summary | 0.244683 | 0.682738 | 54.25% | 53.48% |
| series_winner / BNLC-original_sum | 0.244980 | 0.683333 | 54.20% | 53.20% |

## All contrasts

Negative favors first procedure. 131072 complete-event draws; original centered-bootstrap/Monte-Carlo-bound/Holm arithmetic is applied within each new complete family. Intervals condition on the fitted recipes.

| Task / family / candidate − reference / loss | Difference | Pointwise 95% interval | Adjusted upper p | Decision |
|---|---:|---:|---:|---|
| map_winner/pool80/BN-mean-BN-sum/brier | +0.000513 | [+0.000166, +0.000865] | 0.370980 | harm unresolved |
| map_winner/pool80/BN-mean-BN-sum/log_loss | +0.001134 | [+0.000266, +0.001994] | 0.807123 | harm unresolved |
| map_winner/pool80/BN-attention-BN-sum/brier | +0.000580 | [+0.000228, +0.000933] | 0.127969 | harm unresolved |
| map_winner/pool80/BN-attention-BN-sum/log_loss | +0.001288 | [+0.000415, +0.002169] | 0.342537 | harm unresolved |
| map_winner/pool80/BN-attention-BN-mean/brier | +0.000068 | [-0.000034, +0.000178] | 1.000000 | harm unresolved |
| map_winner/pool80/BN-attention-BN-mean/log_loss | +0.000154 | [-0.000076, +0.000400] | 1.000000 | harm unresolved |
| map_winner/pool80/BN-sum-BN-summary/brier | -0.000213 | [-0.000724, +0.000290] | 1.000000 | benefit unresolved |
| map_winner/pool80/BN-sum-BN-summary/log_loss | -0.000137 | [-0.001274, +0.000984] | 1.000000 | benefit unresolved |
| map_winner/pool80/BN-attention-BN-summary/brier | +0.000367 | [-0.000235, +0.000978] | 1.000000 | harm unresolved |
| map_winner/pool80/BN-attention-BN-summary/log_loss | +0.001151 | [-0.000227, +0.002562] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-mean-BNC-sum/brier | +0.000258 | [-0.000108, +0.000616] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-mean-BNC-sum/log_loss | +0.000699 | [-0.000184, +0.001573] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-attention-BNC-sum/brier | +0.000191 | [-0.000151, +0.000529] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-attention-BNC-sum/log_loss | +0.000520 | [-0.000330, +0.001363] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-attention-BNC-mean/brier | -0.000067 | [-0.000188, +0.000051] | 1.000000 | benefit unresolved |
| map_winner/pool80/BNC-attention-BNC-mean/log_loss | -0.000179 | [-0.000445, +0.000084] | 1.000000 | benefit unresolved |
| map_winner/pool80/BNC-sum-BNC-summary/brier | +0.000153 | [-0.000227, +0.000543] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-sum-BNC-summary/log_loss | +0.000739 | [-0.000164, +0.001681] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-attention-BNC-summary/brier | +0.000345 | [-0.000144, +0.000840] | 1.000000 | harm unresolved |
| map_winner/pool80/BNC-attention-BNC-summary/log_loss | +0.001259 | [+0.000053, +0.002501] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-mean-BNL-sum/brier | +0.000453 | [-0.000093, +0.001004] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-mean-BNL-sum/log_loss | +0.000793 | [-0.000525, +0.002094] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-sum/brier | +0.000517 | [-0.000036, +0.001073] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-sum/log_loss | +0.000909 | [-0.000413, +0.002219] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-mean/brier | +0.000064 | [-0.000005, +0.000132] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-mean/log_loss | +0.000117 | [-0.000025, +0.000258] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-sum-BNL-summary/brier | -0.000009 | [-0.000473, +0.000446] | 1.000000 | benefit unresolved |
| map_winner/pool80/BNL-sum-BNL-summary/log_loss | +0.000482 | [-0.000572, +0.001541] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-summary/brier | +0.000507 | [-0.000145, +0.001164] | 1.000000 | harm unresolved |
| map_winner/pool80/BNL-attention-BNL-summary/log_loss | +0.001391 | [-0.000094, +0.002897] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-mean-BNLC-sum/brier | +0.000035 | [-0.000534, +0.000601] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-mean-BNLC-sum/log_loss | -0.000025 | [-0.001353, +0.001296] | 1.000000 | benefit unresolved |
| map_winner/pool80/BNLC-attention-BNLC-sum/brier | +0.000081 | [-0.000486, +0.000651] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-attention-BNLC-sum/log_loss | +0.000075 | [-0.001252, +0.001400] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-attention-BNLC-mean/brier | +0.000046 | [-0.000050, +0.000150] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-attention-BNLC-mean/log_loss | +0.000100 | [-0.000115, +0.000328] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-sum-BNLC-summary/brier | +0.000163 | [-0.000440, +0.000758] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-sum-BNLC-summary/log_loss | +0.000568 | [-0.000751, +0.001904] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-attention-BNLC-summary/brier | +0.000245 | [-0.000278, +0.000763] | 1.000000 | harm unresolved |
| map_winner/pool80/BNLC-attention-BNLC-summary/log_loss | +0.000643 | [-0.000524, +0.001805] | 1.000000 | harm unresolved |
| map_winner/channel64/BNC-sum-BN-sum/brier | +0.000458 | [-0.000019, +0.000945] | 1.000000 | harm unresolved |
| map_winner/channel64/BNC-sum-BN-sum/log_loss | +0.001260 | [+0.000177, +0.002354] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-sum-BNL-sum/brier | +0.000286 | [-0.000255, +0.000844] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-sum-BNL-sum/log_loss | +0.000607 | [-0.000622, +0.001849] | 1.000000 | harm unresolved |
| map_winner/channel64/BNL-sum-BN-sum/brier | -0.000096 | [-0.000636, +0.000422] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-sum-BN-sum/log_loss | -0.000204 | [-0.001381, +0.000936] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-sum-BNC-sum/brier | -0.000269 | [-0.000892, +0.000342] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-sum-BNC-sum/log_loss | -0.000856 | [-0.002220, +0.000474] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNC-mean-BN-mean/brier | +0.000204 | [-0.000297, +0.000707] | 1.000000 | harm unresolved |
| map_winner/channel64/BNC-mean-BN-mean/log_loss | +0.000824 | [-0.000311, +0.001960] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-mean-BNL-mean/brier | -0.000132 | [-0.000575, +0.000303] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-mean-BNL-mean/log_loss | -0.000211 | [-0.001230, +0.000779] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-mean-BN-mean/brier | -0.000156 | [-0.000593, +0.000278] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-mean-BN-mean/log_loss | -0.000545 | [-0.001538, +0.000443] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-mean-BNC-mean/brier | -0.000492 | [-0.000927, -0.000054] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-mean-BNC-mean/log_loss | -0.001580 | [-0.002608, -0.000553] | 0.181644 | benefit unresolved |
| map_winner/channel64/BNC-attention-BN-attention/brier | +0.000069 | [-0.000432, +0.000565] | 1.000000 | harm unresolved |
| map_winner/channel64/BNC-attention-BN-attention/log_loss | +0.000492 | [-0.000646, +0.001630] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-attention-BNL-attention/brier | -0.000150 | [-0.000568, +0.000260] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-attention-BNL-attention/log_loss | -0.000228 | [-0.001184, +0.000709] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-attention-BN-attention/brier | -0.000160 | [-0.000651, +0.000328] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-attention-BN-attention/log_loss | -0.000582 | [-0.001695, +0.000508] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-attention-BNC-attention/brier | -0.000379 | [-0.000808, +0.000050] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-attention-BNC-attention/log_loss | -0.001301 | [-0.002328, -0.000290] | 0.703520 | benefit unresolved |
| map_winner/channel64/BNC-summary-BN-summary/brier | +0.000091 | [-0.000501, +0.000687] | 1.000000 | harm unresolved |
| map_winner/channel64/BNC-summary-BN-summary/log_loss | +0.000384 | [-0.000913, +0.001682] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-summary-BNL-summary/brier | +0.000113 | [-0.000488, +0.000720] | 1.000000 | harm unresolved |
| map_winner/channel64/BNLC-summary-BNL-summary/log_loss | +0.000521 | [-0.000803, +0.001835] | 1.000000 | harm unresolved |
| map_winner/channel64/BNL-summary-BN-summary/brier | -0.000301 | [-0.000755, +0.000153] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNL-summary-BN-summary/log_loss | -0.000822 | [-0.001808, +0.000173] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-summary-BNC-summary/brier | -0.000279 | [-0.000741, +0.000173] | 1.000000 | benefit unresolved |
| map_winner/channel64/BNLC-summary-BNC-summary/log_loss | -0.000686 | [-0.001708, +0.000312] | 1.000000 | benefit unresolved |
| map_winner/projection4/BNLC-sum-BNLC-original_sum/brier | +0.000211 | [-0.000043, +0.000458] | 0.160379 | harm unresolved |
| map_winner/projection4/BNLC-sum-BNLC-original_sum/log_loss | +0.000698 | [+0.000091, +0.001299] | 0.099321 | harm unresolved |
| series_winner/pool80/BN-mean-BN-sum/brier | +0.000289 | [-0.000353, +0.000996] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-mean-BN-sum/log_loss | +0.000636 | [-0.000760, +0.002164] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-attention-BN-sum/brier | +0.000190 | [-0.000372, +0.000809] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-attention-BN-sum/log_loss | +0.000429 | [-0.000790, +0.001782] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-attention-BN-mean/brier | -0.000099 | [-0.000225, +0.000024] | 1.000000 | benefit unresolved |
| series_winner/pool80/BN-attention-BN-mean/log_loss | -0.000207 | [-0.000475, +0.000053] | 1.000000 | benefit unresolved |
| series_winner/pool80/BN-sum-BN-summary/brier | +0.000250 | [-0.000172, +0.000663] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-sum-BN-summary/log_loss | +0.000556 | [-0.000396, +0.001483] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-attention-BN-summary/brier | +0.000440 | [-0.000185, +0.001062] | 1.000000 | harm unresolved |
| series_winner/pool80/BN-attention-BN-summary/log_loss | +0.000985 | [-0.000398, +0.002367] | 1.000000 | harm unresolved |
| series_winner/pool80/BNC-mean-BNC-sum/brier | -0.000218 | [-0.000494, +0.000051] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNC-mean-BNC-sum/log_loss | -0.000493 | [-0.001120, +0.000120] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNC-attention-BNC-sum/brier | -0.000031 | [-0.000315, +0.000254] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNC-attention-BNC-sum/log_loss | -0.000082 | [-0.000723, +0.000555] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNC-attention-BNC-mean/brier | +0.000187 | [+0.000090, +0.000291] | 0.045806 | harm supported |
| series_winner/pool80/BNC-attention-BNC-mean/log_loss | +0.000411 | [+0.000194, +0.000643] | 0.059501 | harm unresolved |
| series_winner/pool80/BNC-sum-BNC-summary/brier | +0.000456 | [-0.000127, +0.001102] | 1.000000 | harm unresolved |
| series_winner/pool80/BNC-sum-BNC-summary/log_loss | +0.001042 | [-0.000266, +0.002504] | 1.000000 | harm unresolved |
| series_winner/pool80/BNC-attention-BNC-summary/brier | +0.000425 | [-0.000144, +0.001018] | 1.000000 | harm unresolved |
| series_winner/pool80/BNC-attention-BNC-summary/log_loss | +0.000960 | [-0.000276, +0.002271] | 1.000000 | harm unresolved |
| series_winner/pool80/BNL-mean-BNL-sum/brier | -0.000440 | [-0.000717, -0.000163] | 0.195065 | benefit unresolved |
| series_winner/pool80/BNL-mean-BNL-sum/log_loss | -0.000936 | [-0.001536, -0.000335] | 0.227169 | benefit unresolved |
| series_winner/pool80/BNL-attention-BNL-sum/brier | -0.000438 | [-0.000722, -0.000155] | 0.239910 | benefit unresolved |
| series_winner/pool80/BNL-attention-BNL-sum/log_loss | -0.000920 | [-0.001542, -0.000311] | 0.293431 | benefit unresolved |
| series_winner/pool80/BNL-attention-BNL-mean/brier | +0.000002 | [-0.000054, +0.000056] | 1.000000 | harm unresolved |
| series_winner/pool80/BNL-attention-BNL-mean/log_loss | +0.000016 | [-0.000107, +0.000137] | 1.000000 | harm unresolved |
| series_winner/pool80/BNL-sum-BNL-summary/brier | +0.000170 | [-0.000832, +0.001124] | 1.000000 | harm unresolved |
| series_winner/pool80/BNL-sum-BNL-summary/log_loss | -0.000331 | [-0.002774, +0.001972] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNL-attention-BNL-summary/brier | -0.000268 | [-0.001317, +0.000726] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNL-attention-BNL-summary/log_loss | -0.001251 | [-0.003814, +0.001129] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-mean-BNLC-sum/brier | +0.000116 | [-0.000307, +0.000543] | 1.000000 | harm unresolved |
| series_winner/pool80/BNLC-mean-BNLC-sum/log_loss | +0.000194 | [-0.000724, +0.001124] | 1.000000 | harm unresolved |
| series_winner/pool80/BNLC-attention-BNLC-sum/brier | -0.000206 | [-0.000526, +0.000113] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-attention-BNLC-sum/log_loss | -0.000482 | [-0.001177, +0.000217] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-attention-BNLC-mean/brier | -0.000323 | [-0.000580, -0.000050] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-attention-BNLC-mean/log_loss | -0.000676 | [-0.001219, -0.000100] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-sum-BNLC-summary/brier | -0.000011 | [-0.000662, +0.000564] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-sum-BNLC-summary/log_loss | -0.000094 | [-0.001627, +0.001235] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-attention-BNLC-summary/brier | -0.000218 | [-0.000748, +0.000271] | 1.000000 | benefit unresolved |
| series_winner/pool80/BNLC-attention-BNLC-summary/log_loss | -0.000576 | [-0.001827, +0.000533] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNC-sum-BN-sum/brier | +0.001479 | [+0.000546, +0.002492] | 0.230414 | harm unresolved |
| series_winner/channel64/BNC-sum-BN-sum/log_loss | +0.003246 | [+0.001196, +0.005498] | 0.260049 | harm unresolved |
| series_winner/channel64/BNLC-sum-BNL-sum/brier | +0.000249 | [-0.000309, +0.000768] | 1.000000 | harm unresolved |
| series_winner/channel64/BNLC-sum-BNL-sum/log_loss | +0.000475 | [-0.000825, +0.001664] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-sum-BN-sum/brier | +0.000682 | [+0.000048, +0.001398] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-sum-BN-sum/log_loss | +0.001498 | [+0.000098, +0.003122] | 1.000000 | harm unresolved |
| series_winner/channel64/BNLC-sum-BNC-sum/brier | -0.000548 | [-0.001273, +0.000136] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNLC-sum-BNC-sum/log_loss | -0.001273 | [-0.002878, +0.000232] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNC-mean-BN-mean/brier | +0.000972 | [+0.000293, +0.001705] | 0.470225 | harm unresolved |
| series_winner/channel64/BNC-mean-BN-mean/log_loss | +0.002117 | [+0.000577, +0.003786] | 0.622443 | harm unresolved |
| series_winner/channel64/BNLC-mean-BNL-mean/brier | +0.000805 | [+0.000260, +0.001359] | 0.281810 | harm unresolved |
| series_winner/channel64/BNLC-mean-BNL-mean/log_loss | +0.001605 | [+0.000367, +0.002860] | 0.684532 | harm unresolved |
| series_winner/channel64/BNL-mean-BN-mean/brier | -0.000047 | [-0.000344, +0.000256] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNL-mean-BN-mean/log_loss | -0.000074 | [-0.000724, +0.000586] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNLC-mean-BNC-mean/brier | -0.000214 | [-0.000630, +0.000191] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNLC-mean-BNC-mean/log_loss | -0.000587 | [-0.001526, +0.000317] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNC-attention-BN-attention/brier | +0.001258 | [+0.000540, +0.002010] | 0.085505 | harm unresolved |
| series_winner/channel64/BNC-attention-BN-attention/log_loss | +0.002736 | [+0.001140, +0.004441] | 0.111681 | harm unresolved |
| series_winner/channel64/BNLC-attention-BNL-attention/brier | +0.000481 | [-0.000080, +0.001061] | 1.000000 | harm unresolved |
| series_winner/channel64/BNLC-attention-BNL-attention/log_loss | +0.000913 | [-0.000358, +0.002203] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-attention-BN-attention/brier | +0.000053 | [-0.000264, +0.000372] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-attention-BN-attention/log_loss | +0.000148 | [-0.000537, +0.000851] | 1.000000 | harm unresolved |
| series_winner/channel64/BNLC-attention-BNC-attention/brier | -0.000723 | [-0.001294, -0.000150] | 0.757339 | benefit unresolved |
| series_winner/channel64/BNLC-attention-BNC-attention/log_loss | -0.001674 | [-0.002939, -0.000429] | 0.597010 | benefit unresolved |
| series_winner/channel64/BNC-summary-BN-summary/brier | +0.001273 | [+0.000487, +0.002106] | 0.181787 | harm unresolved |
| series_winner/channel64/BNC-summary-BN-summary/log_loss | +0.002760 | [+0.001001, +0.004628] | 0.223370 | harm unresolved |
| series_winner/channel64/BNLC-summary-BNL-summary/brier | +0.000431 | [-0.000443, +0.001349] | 1.000000 | harm unresolved |
| series_winner/channel64/BNLC-summary-BNL-summary/log_loss | +0.000239 | [-0.002001, +0.002496] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-summary-BN-summary/brier | +0.000761 | [+0.000044, +0.001555] | 1.000000 | harm unresolved |
| series_winner/channel64/BNL-summary-BN-summary/log_loss | +0.002384 | [+0.000574, +0.004434] | 0.885658 | harm unresolved |
| series_winner/channel64/BNLC-summary-BNC-summary/brier | -0.000081 | [-0.000690, +0.000600] | 1.000000 | benefit unresolved |
| series_winner/channel64/BNLC-summary-BNC-summary/log_loss | -0.000137 | [-0.001518, +0.001464] | 1.000000 | benefit unresolved |
| series_winner/projection4/BNLC-sum-BNLC-original_sum/brier | -0.000308 | [-0.000649, -0.000027] | 0.160379 | benefit unresolved |
| series_winner/projection4/BNLC-sum-BNLC-original_sum/log_loss | -0.000690 | [-0.001492, -0.000031] | 0.160379 | benefit unresolved |

Machine-readable source: [`player-results/results.json`](../../figure-reconstruction/package/data/revision/player-results/results.json). Fitting receipts retain parameter counts, epoch/batch telemetry, failures and every checkpoint.
