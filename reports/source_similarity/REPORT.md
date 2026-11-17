# Source-project similarity selection

## Outcome

This fixed experiment compares all six sources, the three closest sources by unlabelled metric distributions, and the mean of all 20 three-source subsets. No outer-target label selected a source, model, subset, or threshold.

- Rows/releases/projects: 8,416 / 25 / 7.
- Prediction model: the unchanged fixed logistic-regression pipeline at threshold 0.5.
- Similarity: source-fitted median imputation and standard scaling; 40-value median/IQR summaries; Euclidean distance.
- The method accesses unlabelled target metrics. Target labels never enter selection.
- The subset reference matches three source projects, not equal training-row counts.
- Existing all-source logistic results reproduced exactly: **True**.
- Protected prior artifacts unchanged: **True**.
- Configuration: `config/source_similarity.json`; protocol: `docs/SOURCE_SIMILARITY_PROTOCOL.md`.

## Selected sources and distances

| Target | Selected sources (distance) | Other sources (distance) |
|---|---|---|
| camel | lucene (1.685), jedit (2.076), velocity (2.571) | synapse (2.669), log4j (3.365), poi (3.536) |
| jedit | camel (2.589), lucene (2.629), velocity (3.048) | synapse (3.661), log4j (3.873), poi (4.303) |
| log4j | velocity (2.231), lucene (2.407), synapse (3.292) | camel (3.293), poi (3.364), jedit (3.507) |
| lucene | velocity (1.545), camel (1.703), jedit (2.165) | log4j (2.396), synapse (2.636), poi (2.865) |
| poi | lucene (2.939), log4j (3.467), velocity (3.603) | camel (3.676), jedit (3.779), synapse (4.524) |
| synapse | lucene (2.620), camel (2.879), jedit (3.113) | velocity (3.173), log4j (3.293), poi (4.376) |
| velocity | lucene (1.555), log4j (2.244), camel (2.531) | jedit (2.642), synapse (3.198), poi (3.491) |

## Per-target ranking results

Random values are mean +/- sample SD with the full min-max range across 20 subsets.

| Target | All AP | Similar AP (diff.) | Random AP | All top-20 R | Similar top-20 R (diff.) | Random top-20 R |
|---|---:|---:|---:|---:|---:|---:|
| camel | 0.268 | 0.252 (-0.016) | 0.273 +/- 0.015 [0.245, 0.300] | 0.281 | 0.265 (-0.016) | 0.286 +/- 0.018 [0.263, 0.327] |
| jedit | 0.349 | 0.284 (-0.066) | 0.350 +/- 0.033 [0.284, 0.395] | 0.413 | 0.363 (-0.050) | 0.416 +/- 0.038 [0.360, 0.475] |
| log4j | 0.743 | 0.751 (+0.008) | 0.735 +/- 0.051 [0.636, 0.802] | 0.292 | 0.308 (+0.015) | 0.286 +/- 0.026 [0.231, 0.312] |
| lucene | 0.727 | 0.719 (-0.009) | 0.712 +/- 0.026 [0.660, 0.746] | 0.267 | 0.274 (+0.007) | 0.274 +/- 0.011 [0.249, 0.295] |
| poi | 0.669 | 0.660 (-0.009) | 0.648 +/- 0.012 [0.626, 0.668] | 0.280 | 0.263 (-0.017) | 0.266 +/- 0.005 [0.252, 0.277] |
| synapse | 0.451 | 0.446 (-0.006) | 0.460 +/- 0.054 [0.375, 0.542] | 0.395 | 0.401 (+0.006) | 0.396 +/- 0.032 [0.321, 0.444] |
| velocity | 0.671 | 0.623 (-0.048) | 0.651 +/- 0.028 [0.613, 0.694] | 0.237 | 0.218 (-0.019) | 0.232 +/- 0.019 [0.204, 0.259] |

## Secondary metrics at 0.5

Random confusion entries are means across the 20 subsets and need not be integers.

| Target | Strategy | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---|---:|---:|---:|---|
| camel | all | 0.284 | 0.269 | 0.276 | 1841 / 381 / 411 / 151 |
| camel | similar | 0.260 | 0.274 | 0.267 | 1784 / 438 / 408 / 154 |
| camel | random mean | 0.291 | 0.331 | 0.273 | 1722.8 / 499.2 / 376.1 / 185.9 |
| jedit | all | 0.299 | 0.531 | 0.382 | 1068 / 378 / 142 / 161 |
| jedit | similar | 0.271 | 0.403 | 0.324 | 1117 / 329 / 181 / 122 |
| jedit | random mean | 0.290 | 0.583 | 0.377 | 987.4 / 458.6 / 126.3 / 176.7 |
| log4j | all | 0.926 | 0.096 | 0.174 | 187 / 2 / 235 / 25 |
| log4j | similar | 0.810 | 0.392 | 0.528 | 165 / 24 / 158 / 102 |
| log4j | random mean | 0.805 | 0.178 | 0.256 | 175.2 / 13.8 / 213.8 / 46.2 |
| lucene | all | 0.920 | 0.105 | 0.189 | 340 / 4 / 392 / 46 |
| lucene | similar | 0.923 | 0.027 | 0.053 | 343 / 1 / 426 / 12 |
| lucene | random mean | 0.840 | 0.207 | 0.289 | 315.7 / 28.3 / 347.4 / 90.7 |
| poi | all | 0.726 | 0.161 | 0.264 | 628 / 43 / 593 / 114 |
| poi | similar | 0.586 | 0.871 | 0.701 | 236 / 435 / 91 / 616 |
| poi | random mean | 0.687 | 0.306 | 0.368 | 557.5 / 113.5 / 490.6 / 216.3 |
| synapse | all | 0.548 | 0.142 | 0.225 | 454 / 19 / 139 / 23 |
| synapse | similar | 0.533 | 0.049 | 0.090 | 466 / 7 / 154 / 8 |
| synapse | random mean | 0.504 | 0.357 | 0.336 | 386.4 / 86.7 / 104.2 / 57.9 |
| velocity | all | 0.738 | 0.084 | 0.152 | 261 / 11 / 336 / 31 |
| velocity | similar | 0.632 | 0.150 | 0.242 | 240 / 32 / 312 / 55 |
| velocity | random mean | 0.674 | 0.141 | 0.214 | 244.1 / 27.9 / 315.2 / 51.8 |

## Training data and runtime

| Target | Strategy | Training rows | Training prevalence | Train s | Predict s |
|---|---|---:|---:|---:|---:|
| camel | all | 5632 | 0.397 | 0.033 | 0.003 |
| camel | similar | 3170 | 0.350 | 0.017 | 0.002 |
| camel | random mean [row range] | 2816.0 [1723, 3909] | 0.410 | 0.014 | 0.002 |
| jedit | all | 6667 | 0.374 | 0.025 | 0.003 |
| jedit | similar | 4205 | 0.325 | 0.018 | 0.002 |
| jedit | random mean [row range] | 3333.5 [1723, 4944] | 0.402 | 0.015 | 0.002 |
| log4j | all | 7967 | 0.319 | 0.045 | 0.002 |
| log4j | similar | 2056 | 0.470 | 0.013 | 0.002 |
| log4j | random mean [row range] | 3983.5 [2056, 5911] | 0.335 | 0.017 | 0.002 |
| lucene | all | 7634 | 0.309 | 0.031 | 0.002 |
| lucene | similar | 5172 | 0.238 | 0.017 | 0.003 |
| lucene | random mean [row range] | 3817.0 [1723, 5911] | 0.327 | 0.017 | 0.002 |
| poi | all | 7038 | 0.297 | 0.028 | 0.003 |
| poi | similar | 1870 | 0.570 | 0.013 | 0.002 |
| poi | random mean [row range] | 3519.0 [1723, 5315] | 0.322 | 0.017 | 0.002 |
| synapse | all | 7781 | 0.339 | 0.027 | 0.003 |
| synapse | similar | 5315 | 0.245 | 0.020 | 0.002 |
| synapse | random mean [row range] | 3890.5 [1870, 5911] | 0.362 | 0.017 | 0.002 |
| velocity | all | 7777 | 0.313 | 0.031 | 0.001 |
| velocity | similar | 4015 | 0.314 | 0.019 | 0.002 |
| velocity | random mean [row range] | 3888.5 [1866, 5911] | 0.330 | 0.019 | 0.002 |

## Equal-weight project averages

The random reference is averaged within each target before targets receive equal weight.

| Strategy | AP | Top-20 recall | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| all_sources | 0.554 +/- 0.194 | 0.309 +/- 0.067 | 0.634 +/- 0.267 | 0.198 +/- 0.159 | 0.237 +/- 0.079 |
| similar_sources | 0.534 +/- 0.206 | 0.299 +/- 0.064 | 0.574 +/- 0.249 | 0.310 +/- 0.290 | 0.315 +/- 0.231 |
| random_subset_mean | 0.547 +/- 0.185 | 0.308 +/- 0.070 | 0.584 +/- 0.228 | 0.300 +/- 0.149 | 0.302 +/- 0.061 |

## Interpretation

- Similar selection changed macro AP by -0.020625 versus all sources, improving 1 targets and regressing 6.
- Similar selection changed macro top-20% recall by -0.010451, improving 3 targets and regressing 4.
- The exhaustive three-source reference changed macro AP by -0.007206 versus all sources. Similar selection should be judged against this size-matched average, not against the best subset.
- Directly versus the three-source reference, similarity selection changed macro AP by -0.013420 and top-20% recall by -0.009267.
- Results are mixed and heterogeneous across projects. They are descriptive; no novelty, statistical significance, or production-readiness claim is made.
- Metric-vector matches were retained. A matching vector alone is not evidence of copied source code.

Full-precision results and all 20 combinations are in the companion CSV files. Large per-record predictions and the complete JSON payload remain under the ignored `artifacts/source_similarity` directory.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.source_similarity
```
