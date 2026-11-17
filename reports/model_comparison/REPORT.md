# Fixed LOPO model comparison

This report executes the protocol frozen in `docs/MODEL_COMPARISON_PROTOCOL.md`. No model setting or threshold was selected from outer-test results. The prior source-selected-threshold experiment remains separate.

- Projects: camel, jedit, log4j, lucene, poi, synapse, velocity.
- Rows/releases: 8,416 / 25.
- Features: all 20 documented input metrics; identifiers and labels excluded.
- Primary seed: 42; random-forest sensitivity seeds: 17, 42, 91.
- Protected frozen artifacts unchanged: **True**.

## Primary per-project results (seed 42)

Top-20% uses an exact ceiling record budget. Confusion matrices are TN / FP / FN / TP at 0.5.

| Project | Model | n | Prev. | AP | Top-20 recall | P | R | F1 | TN / FP / FN / TP | Train s | Predict s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| camel | dummy_prior | 2784 | 0.202 | 0.202 | 0.144 | 0.000 | 0.000 | 0.000 | 2222 / 0 / 562 / 0 | 0.108 | 0.005 |
| camel | logistic_regression | 2784 | 0.202 | 0.268 | 0.281 | 0.284 | 0.269 | 0.276 | 1841 / 381 / 411 / 151 | 0.095 | 0.008 |
| camel | random_forest | 2784 | 0.202 | 0.278 | 0.297 | 0.296 | 0.372 | 0.329 | 1724 / 498 / 353 / 209 | 5.966 | 0.161 |
| camel | hist_gradient_boosting | 2784 | 0.202 | 0.260 | 0.265 | 0.259 | 0.349 | 0.297 | 1660 / 562 / 366 / 196 | 9.698 | 0.047 |
| jedit | dummy_prior | 1749 | 0.173 | 0.173 | 0.300 | 0.000 | 0.000 | 0.000 | 1446 / 0 / 303 / 0 | 0.029 | 0.004 |
| jedit | logistic_regression | 1749 | 0.173 | 0.349 | 0.413 | 0.299 | 0.531 | 0.382 | 1068 / 378 / 142 / 161 | 0.081 | 0.006 |
| jedit | random_forest | 1749 | 0.173 | 0.367 | 0.446 | 0.323 | 0.650 | 0.432 | 1033 / 413 / 106 / 197 | 6.608 | 0.127 |
| jedit | hist_gradient_boosting | 1749 | 0.173 | 0.348 | 0.409 | 0.291 | 0.650 | 0.402 | 965 / 481 / 106 / 197 | 9.820 | 0.164 |
| log4j | dummy_prior | 449 | 0.579 | 0.579 | 0.108 | 0.000 | 0.000 | 0.000 | 189 / 0 / 260 / 0 | 0.038 | 0.003 |
| log4j | logistic_regression | 449 | 0.579 | 0.743 | 0.292 | 0.926 | 0.096 | 0.174 | 187 / 2 / 235 / 25 | 0.099 | 0.005 |
| log4j | random_forest | 449 | 0.579 | 0.764 | 0.304 | 0.917 | 0.169 | 0.286 | 185 / 4 / 216 / 44 | 6.432 | 0.081 |
| log4j | hist_gradient_boosting | 449 | 0.579 | 0.731 | 0.277 | 0.845 | 0.188 | 0.308 | 180 / 9 / 211 / 49 | 8.918 | 0.022 |
| lucene | dummy_prior | 782 | 0.560 | 0.560 | 0.167 | 0.000 | 0.000 | 0.000 | 344 / 0 / 438 / 0 | 0.031 | 0.004 |
| lucene | logistic_regression | 782 | 0.560 | 0.727 | 0.267 | 0.920 | 0.105 | 0.189 | 340 / 4 / 392 / 46 | 0.106 | 0.005 |
| lucene | random_forest | 782 | 0.560 | 0.723 | 0.281 | 0.855 | 0.162 | 0.273 | 332 / 12 / 367 / 71 | 4.506 | 0.073 |
| lucene | hist_gradient_boosting | 782 | 0.560 | 0.671 | 0.258 | 0.704 | 0.228 | 0.345 | 302 / 42 / 338 / 100 | 6.903 | 0.018 |
| poi | dummy_prior | 1378 | 0.513 | 0.513 | 0.214 | 0.000 | 0.000 | 0.000 | 671 / 0 / 707 / 0 | 0.101 | 0.004 |
| poi | logistic_regression | 1378 | 0.513 | 0.669 | 0.280 | 0.726 | 0.161 | 0.264 | 628 / 43 / 593 / 114 | 0.172 | 0.005 |
| poi | random_forest | 1378 | 0.513 | 0.648 | 0.283 | 0.738 | 0.338 | 0.464 | 586 / 85 / 468 / 239 | 4.960 | 0.115 |
| poi | hist_gradient_boosting | 1378 | 0.513 | 0.636 | 0.264 | 0.655 | 0.328 | 0.437 | 549 / 122 / 475 / 232 | 8.195 | 0.025 |
| synapse | dummy_prior | 635 | 0.255 | 0.255 | 0.074 | 0.000 | 0.000 | 0.000 | 473 / 0 / 162 / 0 | 0.092 | 0.007 |
| synapse | logistic_regression | 635 | 0.255 | 0.451 | 0.395 | 0.548 | 0.142 | 0.225 | 454 / 19 / 139 / 23 | 0.358 | 0.037 |
| synapse | random_forest | 635 | 0.255 | 0.391 | 0.346 | 0.494 | 0.247 | 0.329 | 432 / 41 / 122 / 40 | 7.770 | 0.209 |
| synapse | hist_gradient_boosting | 635 | 0.255 | 0.312 | 0.302 | 0.400 | 0.222 | 0.286 | 419 / 54 / 126 / 36 | 8.463 | 0.017 |
| velocity | dummy_prior | 639 | 0.574 | 0.574 | 0.308 | 0.000 | 0.000 | 0.000 | 272 / 0 / 367 / 0 | 0.127 | 0.010 |
| velocity | logistic_regression | 639 | 0.574 | 0.671 | 0.237 | 0.738 | 0.084 | 0.152 | 261 / 11 / 336 / 31 | 0.402 | 0.036 |
| velocity | random_forest | 639 | 0.574 | 0.643 | 0.234 | 0.712 | 0.114 | 0.197 | 255 / 17 / 325 / 42 | 6.302 | 0.062 |
| velocity | hist_gradient_boosting | 639 | 0.574 | 0.641 | 0.229 | 0.685 | 0.136 | 0.227 | 249 / 23 / 317 / 50 | 3.927 | 0.050 |

## Equal-weight project averages

Each project counts once. The `+/-` value is sample standard deviation across the seven projects.

| Model | AP | Top-20 recall | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| dummy_prior | 0.408 +/- 0.188 | 0.188 +/- 0.091 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| logistic_regression | 0.554 +/- 0.194 | 0.309 +/- 0.067 | 0.634 +/- 0.267 | 0.198 +/- 0.159 | 0.237 +/- 0.079 |
| random_forest | 0.545 +/- 0.194 | 0.313 +/- 0.067 | 0.619 +/- 0.250 | 0.293 +/- 0.184 | 0.330 +/- 0.092 |
| hist_gradient_boosting | 0.514 +/- 0.198 | 0.286 +/- 0.058 | 0.548 +/- 0.229 | 0.300 +/- 0.171 | 0.329 +/- 0.072 |

## Per-project differences from logistic regression

Positive values favor the named model; negative values favor logistic regression.

| Project | Model | AP diff. | Top-20 recall diff. | Precision diff. | Recall diff. | F1 diff. |
|---|---|---:|---:|---:|---:|---:|
| camel | dummy_prior | -0.066 | -0.137 | -0.284 | -0.269 | -0.276 |
| camel | random_forest | +0.010 | +0.016 | +0.012 | +0.103 | +0.053 |
| camel | hist_gradient_boosting | -0.008 | -0.016 | -0.025 | +0.080 | +0.021 |
| jedit | dummy_prior | -0.176 | -0.112 | -0.299 | -0.531 | -0.382 |
| jedit | random_forest | +0.017 | +0.033 | +0.024 | +0.119 | +0.049 |
| jedit | hist_gradient_boosting | -0.002 | -0.003 | -0.008 | +0.119 | +0.019 |
| log4j | dummy_prior | -0.164 | -0.185 | -0.926 | -0.096 | -0.174 |
| log4j | random_forest | +0.021 | +0.012 | -0.009 | +0.073 | +0.111 |
| log4j | hist_gradient_boosting | -0.012 | -0.015 | -0.081 | +0.092 | +0.134 |
| lucene | dummy_prior | -0.167 | -0.100 | -0.920 | -0.105 | -0.189 |
| lucene | random_forest | -0.005 | +0.014 | -0.065 | +0.057 | +0.084 |
| lucene | hist_gradient_boosting | -0.057 | -0.009 | -0.216 | +0.123 | +0.156 |
| poi | dummy_prior | -0.156 | -0.066 | -0.726 | -0.161 | -0.264 |
| poi | random_forest | -0.021 | +0.003 | +0.012 | +0.177 | +0.200 |
| poi | hist_gradient_boosting | -0.033 | -0.016 | -0.071 | +0.167 | +0.173 |
| synapse | dummy_prior | -0.196 | -0.321 | -0.548 | -0.142 | -0.225 |
| synapse | random_forest | -0.060 | -0.049 | -0.054 | +0.105 | +0.104 |
| synapse | hist_gradient_boosting | -0.139 | -0.093 | -0.148 | +0.080 | +0.060 |
| velocity | dummy_prior | -0.097 | +0.071 | -0.738 | -0.084 | -0.152 |
| velocity | random_forest | -0.029 | -0.003 | -0.026 | +0.030 | +0.046 |
| velocity | hist_gradient_boosting | -0.030 | -0.008 | -0.053 | +0.052 | +0.076 |

## Seed sensitivity and runtime

Random-forest seeds are repeated measurements of the same seven folds, not independent projects. The table reports variation among the equal-project macro result at each seed.

| Metric | Mean across seed macros | SD | Min | Max |
|---|---:|---:|---:|---:|
| average_precision | 0.544635 | 0.001192 | 0.543432 | 0.545815 |
| top_20_percent_recall | 0.314269 | 0.001228 | 0.312895 | 0.315260 |
| precision | 0.616019 | 0.003605 | 0.612074 | 0.619143 |
| recall | 0.285371 | 0.008049 | 0.277167 | 0.293255 |
| f1 | 0.322221 | 0.007002 | 0.316171 | 0.329890 |

| Model | Seed | Fits | Train total s | Predict total s |
|---|---:|---:|---:|---:|
| dummy_prior | 42 | 7 | 0.526 | 0.037 |
| hist_gradient_boosting | 42 | 7 | 55.923 | 0.344 |
| logistic_regression | 42 | 7 | 1.314 | 0.101 |
| random_forest | 17 | 7 | 37.870 | 0.840 |
| random_forest | 42 | 7 | 42.544 | 0.829 |
| random_forest | 91 | 7 | 48.967 | 0.719 |

## Duplicate-overlap audit

No rows were removed. An exact metric-vector match means the 20 predictors match; it does not prove source-code duplication.

| Held-out project | Exact IDs | Same class names | Metric-vector matches n (%) | Conflicting-target matches |
|---|---:|---:|---:|---:|
| camel | 0 | 0 | 161 (5.8%) | 60 |
| jedit | 0 | 0 | 36 (2.1%) | 13 |
| log4j | 0 | 0 | 18 (4.0%) | 12 |
| lucene | 0 | 0 | 16 (2.0%) | 8 |
| poi | 0 | 0 | 27 (2.0%) | 14 |
| synapse | 0 | 0 | 23 (3.6%) | 10 |
| velocity | 0 | 0 | 29 (4.5%) | 23 |

## Interpretation

- random_forest macro AP was 0.544657, a -0.009513 difference from logistic regression, with 3 project gains and 4 regressions.
- hist_gradient_boosting macro AP was 0.513979, a -0.040192 difference from logistic regression, with 0 project gains and 7 regressions.
- Threshold metrics describe the predeclared 0.5 operating point and can move differently from ranking metrics.
- The seven projects are heterogeneous and the observed differences are descriptive. No statistical significance or production-readiness claim is made.
- Per-record predictions, every seed run, checksums, timings, and full-precision values are in the companion CSV and JSON files.
