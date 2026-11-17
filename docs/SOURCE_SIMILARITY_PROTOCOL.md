# Source-project similarity-selection protocol

This protocol was frozen before running `source_project_similarity_selection_lopo_v1`. It asks whether choosing source projects with similar metric distributions improves cross-project defect ranking relative to all sources or equally sized source-project subsets. It does not choose a new prediction model, tune a model or threshold, or inspect outer-target labels during source selection.

## Data, outer splits, features, and target

The experiment uses the same 8,416 rows, 25 releases, and seven leave-one-project-out folds as the existing fixed model comparison:

- Camel: 1.0, 1.2, 1.4, 1.6
- JEdit: 3.2, 4.0, 4.1, 4.2, 4.3 source files (normalized release values are recorded in results)
- Log4j: 1.0, 1.1, 1.2
- Lucene: 2.0, 2.2, 2.4
- POI: 1.5, 2.0, 2.5, 3.0
- Synapse: 1.0, 1.1, 1.2
- Velocity: 1.4, 1.5, 1.6

Every outer fold excludes all releases and rows of its target project from prediction training. Only the original 20 numeric metrics are used: `wmc`, `dit`, `noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `max_cc`, and `avg_cc`. Identifiers (`project`, `release`, `class_name`), the source label `bug_count`, and the derived target `defective` are excluded. The target remains `defective = 1` when `bug_count > 0`, otherwise 0.

## Predeclared strategies

For each target project, the prediction model is fitted under three training strategies:

1. **All sources:** all six non-target projects.
2. **Similar sources:** the three closest non-target projects under the rule below.
3. **Random-subset reference:** every one of the `C(6, 3) = 20` unique three-project combinations.

The random-subset result for a target is the arithmetic mean across all 20 combinations, never the best combination. Sample standard deviation, minimum, and maximum describe the 20 results. Subsets are repeated training conditions within one target, not independent target projects. This reference matches the number of source projects (three), not training-row counts; project sizes differ.

## Similarity rule and access to unlabelled target data

For each outer split, a separate similarity-preprocessing pipeline is fitted once on the six source projects' 20 metric columns only:

1. Median imputation with `keep_empty_features=True`.
2. Standard scaling fitted to the imputed source rows.
3. Transform the source metrics and the target project's unlabelled metric rows using those source-fitted settings.
4. Represent each project by a 40-value vector: the 20 per-feature medians in schema order followed by the 20 interquartile ranges (`75th percentile - 25th percentile`) in schema order.
5. Compute Euclidean distance from the target summary to each source-project summary.
6. Sort by distance ascending and then project identifier ascending; select the first three.

Target metric distributions therefore influence source selection. This is an approach with access to unlabelled target data, not a source-only selection method. Target `bug_count` and `defective` values are not accepted by or read in the similarity calculation.

All-missing and constant features are retained so every summary has exactly 40 values. If a metric is entirely missing across the outer source rows, the imputer's explicit retained-column fill is zero; standard scaling then records zero variance and uses unit scale. A source-constant metric likewise receives unit scale. Every source project has the same standardized value in either case, so any target contribution is identical in every source distance and cannot change source ordering. If the target metric is entirely missing while sources are not, its values become the pooled source median after imputation and its target IQR is zero. No distance is non-finite; the implementation records source-all-missing and source-constant feature names for each fold.

## Prediction training and evaluation

After a source set is fixed, a new prediction pipeline is constructed and fitted only on rows from that source set. Similarity preprocessing is never reused for prediction. The prediction pipeline is the existing fixed baseline: median imputation with missing indicators, standard scaling, and `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)`. All-source, similar-source, and each of the 20 subset fits use the same configuration.

Primary metrics are average precision and top-20% recall. Top-20% reviews exactly `ceil(0.20 * target rows)` records, sorted by score descending with ties broken by release ascending, class name ascending, and original order ascending. Secondary metrics are precision, recall, F1, and TN/FP/FN/TP at the fixed inclusive threshold `score >= 0.5`. No threshold is selected here.

Each target reports source distances and selection, training rows and prevalence, metrics and runtime for all-source and similar-source fits, and mean/sample-SD/range across the 20 subset fits. Differences are similar minus all and random-subset mean minus all. Equal-weight summaries first average subset results within each target, then average the seven target-level values; subsets never increase the project count. Training and inference wall-clock times are descriptive.

Existing metric-vector matches are retained. A matching 20-number vector alone is not treated as proof of duplicated source code. The existing cross-project overlap audit is included without changing any row or evaluation rule.

The experiment is descriptive. It does not claim novelty, statistical significance, production readiness, or a deployed-model change.
