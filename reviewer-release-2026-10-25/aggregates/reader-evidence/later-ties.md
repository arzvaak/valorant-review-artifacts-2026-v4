## S23. Later-cohort exact-tie diagnostic

This descriptive audit reads the original frozen later-cohort forecasts and targets; it performs no inference, fitting, calibration or selection. The original finite-target masks retain 1,613 maps and 734 series across 23 events. Ordinary accuracy exactly reproduces the saved decisions, including residual raw-probability thresholds. Half-credit replaces only decisions whose saved probability equals exactly 0.5 under that procedure’s declared calibration policy by credit 0.5; all other decisions remain unchanged. This is a diagnostic, not a replacement decision policy. No probability or probability-loss result changes. Zero ties have undefined tied-subset class balance. Event accuracy weights each of the original 23 events equally; row accuracy weights targets equally. The audit does not infer why a particular rating or mixture produces an exact tie.

| Task / procedure | Rows | Exact ties | Team A wins among ties | Tied Team A fraction | Ordinary row / event accuracy (%) | Half-credit row / event accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|
| map / dynamic strength | 1613 | 55 | 28 | 0.509091 | 50.651 / 52.746 | 50.682 / 52.849 |
| map / online elo k20 | 1613 | 1115 | 580 | 0.520179 | 51.395 / 51.351 | 52.790 / 54.381 |
| map / training prior | 1613 | 0 | 0 | undefined | 52.139 / 53.892 | 52.139 / 53.892 |
| map / chance | 1613 | 1613 | 841 | 0.521389 | 47.861 / 46.108 | 50.000 / 50.000 |
| map / original residual | 1613 | 0 | 0 | undefined | 62.678 / 58.382 | 62.678 / 58.382 |
| map / rich residual | 1613 | 0 | 0 | undefined | 62.678 / 59.701 | 62.678 / 59.701 |
| series / dynamic strength | 734 | 518 | 274 | 0.528958 | 47.956 / 50.124 | 50.000 / 54.292 |
| series / online elo k20 | 734 | 519 | 275 | 0.529865 | 49.728 / 49.943 | 51.839 / 54.835 |
| series / training prior | 734 | 0 | 0 | undefined | 52.861 / 55.494 | 52.861 / 55.494 |
| series / chance | 734 | 734 | 388 | 0.528610 | 47.139 / 44.506 | 50.000 / 50.000 |
| series / original residual | 734 | 0 | 0 | undefined | 49.183 / 45.409 | 49.183 / 45.409 |
| series / rich residual | 734 | 0 | 0 | undefined | 49.864 / 47.522 | 49.864 / 47.522 |
