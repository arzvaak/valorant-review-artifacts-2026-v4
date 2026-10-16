# Complete additional secondary metric inventory

Descriptive point estimates condition on saved fits. AUC and ECE use equal-event weights; balanced accuracy gives the two target classes equal row weight. Canonical-orientation intercept/slope diagnose original probabilities without modifying them. No new hypothesis family or selection is introduced. Coefficient intervals for selected main procedures appear elsewhere; none is implied for these additional points. Rating, player and removal decisions use probability greater than 0.5; feedback rows preserve their saved decisions. Frozen 2 h and 6 h rating histories have different admission delays; online arms use 6 h throughout. Labels below abbreviate map-winner and series-winner tasks.

## Map / ratings

Rows/events: 55,943/1,049.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| 6h / online raw / elo k20 | 0.60356 | 0.63250 | 0.09080 | 0.96239 | 0.02040 / 0.02136 / 0.02334 |
| 6h / online raw / glicko2 | 0.61538 | 0.64567 | 0.11411 | 0.46015 | 0.08461 / 0.08461 / 0.08515 |
| 6h / online raw / player gaussian | 0.62178 | 0.66086 | 0.14151 | 0.39370 | 0.11003 / 0.11003 / 0.11112 |
| 6h / online calibrated / elo k20 | 0.60356 | 0.62976 | 0.10027 | 0.82901 | 0.02684 / 0.02684 / 0.03125 |
| 6h / online calibrated / glicko2 | 0.61538 | 0.63799 | 0.13304 | 0.75230 | 0.03283 / 0.03557 / 0.03593 |
| 6h / online calibrated / player gaussian | 0.62178 | 0.65374 | 0.15533 | 0.71998 | 0.04095 / 0.04095 / 0.04193 |
| 6h / frozen raw / elo k20 | 0.54567 | 0.56767 | 0.15081 | 0.51858 | 0.04808 / 0.04808 / 0.04819 |
| 6h / frozen raw / glicko2 | 0.54491 | 0.57009 | 0.13102 | 0.24066 | 0.12029 / 0.12029 / 0.12400 |
| 6h / frozen raw / player gaussian | 0.58425 | 0.62224 | 0.16829 | 0.26100 | 0.13821 / 0.13821 / 0.14230 |
| 6h / frozen calibrated / elo k20 | 0.54567 | 0.56410 | 0.16946 | 0.51152 | 0.04757 / 0.04757 / 0.04758 |
| 6h / frozen calibrated / glicko2 | 0.54491 | 0.56322 | 0.17261 | 0.38503 | 0.05139 / 0.05139 / 0.05139 |
| 6h / frozen calibrated / player gaussian | 0.58425 | 0.61696 | 0.18523 | 0.54113 | 0.04974 / 0.05508 / 0.05561 |
| 2h / frozen raw / elo k20 | 0.54571 | 0.56768 | 0.15089 | 0.51843 | 0.04772 / 0.04772 / 0.04810 |
| 2h / frozen raw / glicko2 | 0.54491 | 0.57009 | 0.13102 | 0.24067 | 0.12029 / 0.12029 / 0.12400 |
| 2h / frozen raw / player gaussian | 0.58425 | 0.62224 | 0.16829 | 0.26100 | 0.13821 / 0.13821 / 0.14230 |
| 2h / frozen calibrated / elo k20 | 0.54571 | 0.56415 | 0.16933 | 0.51303 | 0.04698 / 0.04698 / 0.04700 |
| 2h / frozen calibrated / glicko2 | 0.54491 | 0.56322 | 0.17261 | 0.38504 | 0.05139 / 0.05139 / 0.05139 |
| 2h / frozen calibrated / player gaussian | 0.58425 | 0.61696 | 0.18523 | 0.54114 | 0.04974 / 0.05508 / 0.05561 |

## Map / players

Rows/events: 55,943/1,049.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| corrected BN sum | 0.57384 | 0.63214 | 0.09826 | 0.75107 | 0.03162 / 0.03162 / 0.03241 |
| corrected BN mean | 0.57334 | 0.62980 | 0.09948 | 0.74827 | 0.03086 / 0.03086 / 0.03086 |
| corrected BN attention | 0.57319 | 0.62963 | 0.09829 | 0.74551 | 0.03094 / 0.03094 / 0.03104 |
| corrected BN summary | 0.57311 | 0.63118 | 0.09690 | 0.75060 | 0.03398 / 0.03398 / 0.03399 |
| corrected BNC sum | 0.57509 | 0.62968 | 0.10254 | 0.74033 | 0.03065 / 0.03065 / 0.03095 |
| corrected BNC mean | 0.57469 | 0.62917 | 0.10061 | 0.74305 | 0.03031 / 0.03031 / 0.03118 |
| corrected BNC attention | 0.57446 | 0.62887 | 0.10094 | 0.74341 | 0.03076 / 0.03076 / 0.03169 |
| corrected BNC summary | 0.57565 | 0.63026 | 0.10315 | 0.76027 | 0.03047 / 0.03047 / 0.03084 |
| corrected BNL sum | 0.57692 | 0.63225 | 0.10463 | 0.75448 | 0.03025 / 0.03025 / 0.03190 |
| corrected BNL mean | 0.57482 | 0.62919 | 0.10253 | 0.76505 | 0.02921 / 0.02921 / 0.03132 |
| corrected BNL attention | 0.57516 | 0.62899 | 0.10356 | 0.76524 | 0.02991 / 0.02991 / 0.03178 |
| corrected BNL summary | 0.57599 | 0.63088 | 0.10172 | 0.77689 | 0.03228 / 0.03228 / 0.03229 |
| corrected BNLC sum | 0.57692 | 0.63018 | 0.10480 | 0.75537 | 0.03256 / 0.03256 / 0.03481 |
| corrected BNLC mean | 0.57626 | 0.62917 | 0.10471 | 0.77643 | 0.02708 / 0.02708 / 0.02854 |
| corrected BNLC attention | 0.57563 | 0.62875 | 0.10546 | 0.77741 | 0.02672 / 0.02672 / 0.02788 |
| corrected BNLC summary | 0.57773 | 0.63104 | 0.10398 | 0.77499 | 0.02849 / 0.02849 / 0.03032 |
| original BNLC sum | 0.57656 | 0.62988 | 0.10379 | 0.77286 | 0.03051 / 0.03051 / 0.03051 |

## Map / removals

Rows/events: 36,624/720.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| full reference | 0.60078 | 0.65146 | 0.04614 | 0.73386 | 0.02481 / 0.02770 / 0.03527 |
| opening | 0.60251 | 0.65267 | 0.04352 | 0.74367 | 0.02402 / 0.02402 / 0.02791 |
| stars | 0.60069 | 0.65149 | 0.04419 | 0.74101 | 0.02277 / 0.02464 / 0.03154 |
| economy | 0.60267 | 0.65232 | 0.04326 | 0.73749 | 0.02496 / 0.02496 / 0.03473 |
| pistols | 0.60271 | 0.65318 | 0.04114 | 0.73140 | 0.02228 / 0.02446 / 0.03894 |
| map side | 0.59992 | 0.65168 | 0.04682 | 0.74851 | 0.02402 / 0.02402 / 0.02723 |
| impact | 0.60221 | 0.65158 | 0.04860 | 0.74519 | 0.02093 / 0.02312 / 0.03376 |
| experience | 0.60192 | 0.65046 | 0.04442 | 0.73895 | 0.02274 / 0.02274 / 0.03065 |
| schedule stability | 0.60447 | 0.65478 | 0.05278 | 0.71133 | 0.02713 / 0.02972 / 0.03697 |
| matchups | 0.59933 | 0.65004 | 0.04783 | 0.72625 | 0.02321 / 0.02563 / 0.03267 |
| coverage | 0.59154 | 0.63335 | 0.06637 | 0.69791 | 0.03302 / 0.03302 / 0.03402 |

## Map / later feedback

Rows/events: 1,613/23.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| rich residual | 0.64532 | 0.68812 | 0.12397 | 0.72238 | 0.07074 / 0.07109 / 0.08802 |
| original residual | 0.64459 | 0.66151 | 0.12247 | 0.48705 | 0.10259 / 0.10306 / 0.11088 |
| dynamic strength | 0.58115 | 0.58873 | 0.17717 | 2.02140 | 0.05222 / 0.05233 / 0.05242 |
| online elo k20 | 0.58422 | 0.58890 | 0.12565 | 1.58436 | 0.04874 / 0.05188 / 0.06022 |
| chance | 0.50000 | 0.50000 | unresolved | unresolved | 0.03892 / 0.03892 / 0.03892 |
| training prior | 0.50000 | 0.46628 | 20.18602 | -75.20243 | 0.02728 / 0.02728 / 0.02728 |

## Series / ratings

Rows/events: 25,186/1,047.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| 6h / online raw / elo k20 | 0.62571 | 0.64888 | 0.12577 | 1.23395 | 0.04105 / 0.04263 / 0.04309 |
| 6h / online raw / glicko2 | 0.64306 | 0.68007 | 0.12370 | 0.57908 | 0.06005 / 0.06005 / 0.06476 |
| 6h / online raw / player gaussian | 0.64136 | 0.68179 | 0.15910 | 0.53477 | 0.06833 / 0.06833 / 0.07136 |
| 6h / online calibrated / elo k20 | 0.62571 | 0.64573 | 0.14659 | 0.68147 | 0.03839 / 0.04154 / 0.04183 |
| 6h / online calibrated / glicko2 | 0.64306 | 0.67253 | 0.14875 | 0.80840 | 0.03809 / 0.03809 / 0.03809 |
| 6h / online calibrated / player gaussian | 0.64136 | 0.67248 | 0.18662 | 0.78446 | 0.04994 / 0.04994 / 0.04994 |
| 6h / frozen raw / elo k20 | 0.55662 | 0.57368 | 0.20261 | 0.68808 | 0.04850 / 0.05165 / 0.05165 |
| 6h / frozen raw / glicko2 | 0.56049 | 0.58629 | 0.17540 | 0.29382 | 0.10698 / 0.10698 / 0.11272 |
| 6h / frozen raw / player gaussian | 0.55785 | 0.58308 | 0.19329 | 0.25046 | 0.12066 / 0.12066 / 0.12360 |
| 6h / frozen calibrated / elo k20 | 0.55662 | 0.57397 | 0.21632 | 0.43569 | 0.05974 / 0.05974 / 0.06034 |
| 6h / frozen calibrated / glicko2 | 0.56049 | 0.58070 | 0.20783 | 0.41706 | 0.06244 / 0.06244 / 0.06323 |
| 6h / frozen calibrated / player gaussian | 0.55785 | 0.57489 | 0.22870 | 0.39086 | 0.06010 / 0.06010 / 0.06194 |
| 2h / frozen raw / elo k20 | 0.55634 | 0.57352 | 0.20257 | 0.68884 | 0.04873 / 0.05164 / 0.05167 |
| 2h / frozen raw / glicko2 | 0.56049 | 0.58629 | 0.17540 | 0.29383 | 0.10698 / 0.10698 / 0.11271 |
| 2h / frozen raw / player gaussian | 0.55845 | 0.58331 | 0.19264 | 0.25053 | 0.12066 / 0.12066 / 0.12499 |
| 2h / frozen calibrated / elo k20 | 0.55634 | 0.57381 | 0.21606 | 0.43727 | 0.05980 / 0.05980 / 0.05981 |
| 2h / frozen calibrated / glicko2 | 0.56049 | 0.58071 | 0.20783 | 0.41706 | 0.06244 / 0.06244 / 0.06323 |
| 2h / frozen calibrated / player gaussian | 0.55845 | 0.57572 | 0.22758 | 0.39907 | 0.05762 / 0.05778 / 0.05950 |

## Series / players

Rows/events: 25,186/1,047.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| corrected BN sum | 0.55250 | 0.56861 | 0.17673 | 0.53832 | 0.05516 / 0.05518 / 0.05693 |
| corrected BN mean | 0.55210 | 0.56695 | 0.18350 | 0.53374 | 0.05614 / 0.05616 / 0.05646 |
| corrected BN attention | 0.55257 | 0.56795 | 0.18185 | 0.53649 | 0.05589 / 0.05592 / 0.05611 |
| corrected BN summary | 0.55504 | 0.57021 | 0.17857 | 0.55486 | 0.05127 / 0.05129 / 0.05381 |
| corrected BNC sum | 0.55138 | 0.56392 | 0.19381 | 0.48958 | 0.06105 / 0.06114 / 0.06114 |
| corrected BNC mean | 0.55201 | 0.56369 | 0.19374 | 0.49956 | 0.05862 / 0.05870 / 0.05993 |
| corrected BNC attention | 0.55149 | 0.56213 | 0.19553 | 0.49310 | 0.05895 / 0.05903 / 0.06045 |
| corrected BNC summary | 0.55252 | 0.56511 | 0.18963 | 0.50658 | 0.05827 / 0.05834 / 0.05899 |
| corrected BNL sum | 0.55342 | 0.56779 | 0.18648 | 0.51961 | 0.05927 / 0.05929 / 0.05929 |
| corrected BNL mean | 0.55231 | 0.56756 | 0.18396 | 0.53850 | 0.05428 / 0.05431 / 0.05431 |
| corrected BNL attention | 0.55289 | 0.56806 | 0.18380 | 0.53771 | 0.05439 / 0.05443 / 0.05488 |
| corrected BNL summary | 0.55460 | 0.57090 | 0.18359 | 0.50167 | 0.05704 / 0.05704 / 0.05708 |
| corrected BNLC sum | 0.55323 | 0.56421 | 0.18872 | 0.51041 | 0.06025 / 0.06029 / 0.06054 |
| corrected BNLC mean | 0.55254 | 0.56353 | 0.19304 | 0.51224 | 0.06040 / 0.06045 / 0.06068 |
| corrected BNLC attention | 0.55342 | 0.56618 | 0.18900 | 0.52205 | 0.05930 / 0.05936 / 0.05936 |
| corrected BNLC summary | 0.55238 | 0.56576 | 0.19012 | 0.50893 | 0.05856 / 0.05865 / 0.05927 |
| original BNLC sum | 0.55216 | 0.56255 | 0.19111 | 0.49879 | 0.06249 / 0.06254 / 0.06276 |

## Series / removals

Rows/events: 16,319/720.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| full reference | 0.55466 | 0.55157 | 0.17822 | 0.43376 | 0.05777 / 0.05778 / 0.06170 |
| opening | 0.55619 | 0.55389 | 0.17635 | 0.46627 | 0.05528 / 0.05528 / 0.06288 |
| stars | 0.55401 | 0.55531 | 0.18099 | 0.44342 | 0.05504 / 0.05504 / 0.06377 |
| economy | 0.55615 | 0.55346 | 0.17944 | 0.44873 | 0.05669 / 0.05670 / 0.06420 |
| pistols | 0.55527 | 0.55638 | 0.17757 | 0.44196 | 0.05152 / 0.05573 / 0.06315 |
| map side | 0.55554 | 0.55662 | 0.17951 | 0.43420 | 0.05093 / 0.05448 / 0.06389 |
| impact | 0.55466 | 0.55566 | 0.17772 | 0.44888 | 0.05493 / 0.05957 / 0.06154 |
| experience | 0.55495 | 0.55279 | 0.17691 | 0.47489 | 0.04985 / 0.04985 / 0.05822 |
| schedule stability | 0.55720 | 0.55913 | 0.16574 | 0.44899 | 0.05385 / 0.05392 / 0.07252 |
| matchups | 0.55506 | 0.55128 | 0.17977 | 0.45165 | 0.05536 / 0.05536 / 0.06300 |
| coverage | 0.54981 | 0.54849 | 0.17455 | 0.49825 | 0.05197 / 0.05197 / 0.05235 |

## Series / later feedback

Rows/events: 734/23.

| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |
|---|---:|---:|---:|---:|---|
| rich residual | 0.57665 | 0.62489 | 0.12599 | 1.47822 | 0.05816 / 0.08097 / 0.08097 |
| original residual | 0.52772 | 0.57641 | 0.16889 | 0.92013 | 0.08168 / 0.08552 / 0.10126 |
| dynamic strength | 0.57519 | 0.62471 | 0.24731 | 2.77360 | 0.07352 / 0.07352 / 0.08166 |
| online elo k20 | 0.58276 | 0.61293 | 0.19488 | 2.22052 | 0.05669 / 0.08998 / 0.09580 |
| chance | 0.50000 | 0.50000 | unresolved | unresolved | 0.05494 / 0.05494 / 0.05494 |
| training prior | 0.50000 | 0.47083 | 17.67197 | -50.65772 | 0.03034 / 0.03034 / 0.03034 |
