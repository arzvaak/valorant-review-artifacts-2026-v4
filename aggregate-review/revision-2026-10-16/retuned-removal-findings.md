# Complete retuned-removal sensitivity

Every reduced condition searches the original selected full parent's architecture grid using the original train/stopping/selection/calibration phases and seeds. Full reference is the original already-searched parent. Original fixed-configuration removals remain in S4. All 160 new packages froze before evaluation labels were read. Eight supported blocks use 36,624 maps and 16,319 series across 720 events; the same four unsupported blocks remain excluded by the original eligibility rule. The new 40 contrasts receive their own 131,072-draw MonteCarlo/Holm family; original decisions are unchanged.

## Absolute calibrated scores

| Task/condition | Event Brier | Event log loss | Row accuracy | Event accuracy |
|---|---:|---:|---:|---:|
| map_winner/full_reference | 0.231405 | 0.655828 | 0.612849 | 0.612761 |
| map_winner/opening | 0.230991 | 0.654827 | 0.614542 | 0.614042 |
| map_winner/stars | 0.231410 | 0.655839 | 0.612959 | 0.615142 |
| map_winner/economy | 0.231231 | 0.655505 | 0.614897 | 0.616607 |
| map_winner/pistols | 0.231055 | 0.655198 | 0.614979 | 0.616540 |
| map_winner/map_side | 0.231208 | 0.655222 | 0.611593 | 0.613087 |
| map_winner/impact | 0.231296 | 0.655474 | 0.614078 | 0.615022 |
| map_winner/experience | 0.231606 | 0.656146 | 0.613914 | 0.616480 |
| map_winner/schedule_stability | 0.230835 | 0.655092 | 0.615744 | 0.614472 |
| map_winner/matchups | 0.231921 | 0.657033 | 0.611484 | 0.612843 |
| map_winner/coverage | 0.235210 | 0.664293 | 0.603675 | 0.601790 |
| series_winner/full_reference | 0.247576 | 0.689135 | 0.553465 | 0.528373 |
| series_winner/opening | 0.247026 | 0.687682 | 0.555059 | 0.529782 |
| series_winner/stars | 0.247358 | 0.688856 | 0.552730 | 0.530315 |
| series_winner/economy | 0.247386 | 0.688530 | 0.554752 | 0.529030 |
| series_winner/pistols | 0.247354 | 0.688652 | 0.555120 | 0.532849 |
| series_winner/map_side | 0.247584 | 0.689027 | 0.555916 | 0.534110 |
| series_winner/impact | 0.247248 | 0.688308 | 0.554752 | 0.533019 |
| series_winner/experience | 0.246900 | 0.687393 | 0.553894 | 0.528615 |
| series_winner/schedule_stability | 0.247036 | 0.688202 | 0.557265 | 0.533015 |
| series_winner/matchups | 0.247338 | 0.688367 | 0.553894 | 0.528027 |
| series_winner/coverage | 0.246692 | 0.686682 | 0.550830 | 0.526181 |

## All paired reduced-minus-full contrasts

Negative loss difference favors the reduced procedure. Pointwise intervals condition on saved fits; a favorable unadjusted estimate does not supersede the complete corrected family.

| Contrast | Difference | Pointwise95% interval | Conservative Holm | Decision |
|---|---:|---|---:|---|
| map_winner/opening/brier | -0.000414 | [-0.000834, -0.000005] | 1.000000 | unresolved |
| map_winner/opening/log_loss | -0.001001 | [-0.001924, -0.000083] | 1.000000 | unresolved |
| map_winner/stars/brier | +0.000005 | [-0.000607, +0.000611] | 1.000000 | unresolved |
| map_winner/stars/log_loss | +0.000011 | [-0.001324, +0.001334] | 1.000000 | unresolved |
| map_winner/economy/brier | -0.000174 | [-0.000563, +0.000210] | 1.000000 | unresolved |
| map_winner/economy/log_loss | -0.000322 | [-0.001180, +0.000536] | 1.000000 | unresolved |
| map_winner/pistols/brier | -0.000350 | [-0.000754, +0.000040] | 1.000000 | unresolved |
| map_winner/pistols/log_loss | -0.000630 | [-0.001527, +0.000257] | 1.000000 | unresolved |
| map_winner/map_side/brier | -0.000197 | [-0.000769, +0.000357] | 1.000000 | unresolved |
| map_winner/map_side/log_loss | -0.000605 | [-0.001860, +0.000623] | 1.000000 | unresolved |
| map_winner/impact/brier | -0.000109 | [-0.000579, +0.000352] | 1.000000 | unresolved |
| map_winner/impact/log_loss | -0.000354 | [-0.001399, +0.000680] | 1.000000 | unresolved |
| map_winner/experience/brier | +0.000201 | [-0.000321, +0.000717] | 1.000000 | unresolved |
| map_winner/experience/log_loss | +0.000318 | [-0.000826, +0.001447] | 1.000000 | unresolved |
| map_winner/schedule_stability/brier | -0.000570 | [-0.001246, +0.000088] | 1.000000 | unresolved |
| map_winner/schedule_stability/log_loss | -0.000735 | [-0.002305, +0.000796] | 1.000000 | unresolved |
| map_winner/matchups/brier | +0.000516 | [+0.000034, +0.000994] | 1.000000 | unresolved |
| map_winner/matchups/log_loss | +0.001205 | [+0.000159, +0.002252] | 0.891049 | unresolved |
| map_winner/coverage/brier | +0.003805 | [+0.001589, +0.005936] | 0.040484 | harm |
| map_winner/coverage/log_loss | +0.008465 | [+0.003431, +0.013410] | 0.046432 | harm |
| series_winner/opening/brier | -0.000550 | [-0.000948, -0.000163] | 0.243077 | unresolved |
| series_winner/opening/log_loss | -0.001453 | [-0.002392, -0.000543] | 0.093353 | unresolved |
| series_winner/stars/brier | -0.000218 | [-0.000830, +0.000400] | 1.000000 | unresolved |
| series_winner/stars/log_loss | -0.000279 | [-0.001685, +0.001158] | 1.000000 | unresolved |
| series_winner/economy/brier | -0.000190 | [-0.000605, +0.000231] | 1.000000 | unresolved |
| series_winner/economy/log_loss | -0.000605 | [-0.001575, +0.000370] | 1.000000 | unresolved |
| series_winner/pistols/brier | -0.000222 | [-0.000770, +0.000291] | 1.000000 | unresolved |
| series_winner/pistols/log_loss | -0.000483 | [-0.001681, +0.000659] | 1.000000 | unresolved |
| series_winner/map_side/brier | +0.000009 | [-0.000570, +0.000554] | 1.000000 | unresolved |
| series_winner/map_side/log_loss | -0.000108 | [-0.001380, +0.001101] | 1.000000 | unresolved |
| series_winner/impact/brier | -0.000328 | [-0.000955, +0.000283] | 1.000000 | unresolved |
| series_winner/impact/log_loss | -0.000827 | [-0.002170, +0.000491] | 1.000000 | unresolved |
| series_winner/experience/brier | -0.000676 | [-0.001158, -0.000213] | 0.217911 | unresolved |
| series_winner/experience/log_loss | -0.001742 | [-0.002865, -0.000663] | 0.100558 | unresolved |
| series_winner/schedule_stability/brier | -0.000540 | [-0.001688, +0.000690] | 1.000000 | unresolved |
| series_winner/schedule_stability/log_loss | -0.000933 | [-0.003524, +0.001939] | 1.000000 | unresolved |
| series_winner/matchups/brier | -0.000238 | [-0.000745, +0.000279] | 1.000000 | unresolved |
| series_winner/matchups/log_loss | -0.000768 | [-0.001888, +0.000378] | 1.000000 | unresolved |
| series_winner/coverage/brier | -0.000883 | [-0.002236, +0.000434] | 1.000000 | unresolved |
| series_winner/coverage/log_loss | -0.002453 | [-0.005617, +0.000633] | 1.000000 | unresolved |

## Real fitting/failure accounting

Packages: 160; fitted packages: 160; retained fallback packages: 0; candidate records: 1110; failed candidate records: 0. Every candidate/checkpoint/failure remains in its fitting directory.
