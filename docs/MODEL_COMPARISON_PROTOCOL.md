# Fixed model-comparison protocol

This protocol was frozen before running `fixed_model_comparison_lopo_v1`. The comparison does not tune models or thresholds against any outer-test result. It is separate from the source-selected-threshold experiment in `artifacts/source_validation` and the nested model-improvement experiment in `artifacts/model_improvement`.

## Data, split, features, and target

The experiment includes all 8,416 rows in the 25 releases pinned by `config/data_manifest.json`:

- Camel: 1.0, 1.2, 1.4, 1.6
- JEdit: 3.2, 4.0, 4.1, 4.2, 4.3 (the normalized source release values are reported in the results)
- Log4j: 1.0, 1.1, 1.2
- Lucene: 2.0, 2.2, 2.4
- POI: 1.5, 2.0, 2.5, 3.0
- Synapse: 1.0, 1.1, 1.2
- Velocity: 1.4, 1.5, 1.6

Each outer fold holds out every release and row of one project and trains on all rows of the other six projects. Every model receives identical outer train/test records. A fresh pipeline is fitted for every model, seed, and held-out project.

The predictors, in fixed order, are all 20 documented metrics: `wmc`, `dit`, `noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `max_cc`, and `avg_cc`. The target is `defective = 1` when `bug_count > 0`, otherwise 0. Identifiers (`project`, `release`, `class_name`), `bug_count`, and `defective` are excluded from predictive features.

## Metrics and ranking

The primary metrics are average precision and top-20% recall. Top-20% means exactly `ceil(0.20 * test rows)` records. Scores are sorted descending; score ties are resolved deterministically by ascending release and then ascending class name. The fixed budget is not expanded at the boundary when scores tie.

Secondary metrics are precision, recall, F1, and the full TN/FP/FN/TP confusion matrix after applying the fixed inclusive rule `score >= 0.5`. No threshold is selected in this experiment.

Project-level results are the unit of comparison. Across-project summaries are arithmetic means with each of the seven projects weighted equally, plus sample standard deviations across those seven project values. Rows and seeds are not treated as independent projects. Confusion matrices are summed only as descriptive totals and are not used for macro averaging.

## Fixed models and preprocessing

Exact machine-readable settings are in `config/model_comparison.json`.

- Dummy prior: median imputation, then `DummyClassifier(strategy="prior")`. It is the ranking reference.
- Logistic regression: median imputation with missing indicators, standard scaling, then `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)`. This deliberately matches the preserved baseline configuration.
- Random forest: median imputation with missing indicators, then 200 trees, maximum depth 12, minimum leaf size 5, square-root feature sampling, no class weighting, and one worker. Scaling is omitted because tree splits are invariant to monotone feature scaling.
- Histogram gradient boosting: the 20 metrics are passed directly to the estimator so its native missing-value routing is retained; there is no imputation or scaling. It uses learning rate 0.1, 150 iterations, 31 maximum leaves, L2 regularization 1.0, no class weighting, and early stopping disabled.

Any imputation and scaling are pipeline steps fitted only on the outer training rows. No preprocessing statistic is learned from the held-out project.

## Seeds, timing, and validity records

Seed 42 is the sole primary comparison result. Random forest is additionally repeated at seeds 17 and 91 because it samples features and bootstrap rows. Its sensitivity report shows per-seed equal-project macro metrics and variation across the three seeds; repeats are not pooled with project folds and do not increase the sample size. Histogram gradient boosting is fitted once at seed 42 because disabling early stopping removes the estimator's only random validation split; logistic regression and dummy prior also have no stochastic sampling in their declared configurations.

Training and prediction wall-clock times are recorded per fitted fold. Per-record files retain identifiers, target, score, fixed-threshold prediction, top-20% selection, held-out project, model, and seed. The report also records dependency versions, SHA-256 checksums of every raw dataset file, a checksum of each outer test-record identity list, and duplicate-overlap audits. Cross-project duplicates are investigated and reported, but no row is silently removed or altered.
