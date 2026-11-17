# Software Defect Prediction: consolidated research report

Generated from validated full-precision experiment files. Displayed values are rounded only after calculations.

## Executive findings

- Logistic regression had the highest macro average precision (AP) among the tested fixed model configurations.
- Similarity selection did not improve ranking overall, either against all six sources or against the mean of all 20 size-matched three-project subsets.
- Higher F1 at threshold 0.5 does not establish improved ranking; AP and top-20% recall are the primary ranking measures.
- Similarity selection accessed unlabelled target metrics. Target labels were excluded from selection and used only for final evaluation.
- These are exploratory comparisons on seven historical Java projects. No novelty, statistical-significance, or production-readiness claim is established.

## Research questions

1. How well does the fixed logistic baseline rank defective classes under seven-project leave-one-project-out evaluation?
2. Do fixed random-forest or histogram-gradient-boosting configurations improve ranking over logistic regression and a dummy-prior reference?
3. Does training on the three projects with the closest unlabelled metric distributions improve ranking over all six sources or the exhaustive three-source reference?
4. How do fixed-threshold classification results differ from ranking results, including the earlier source-validated threshold experiment?

## Dataset, features, and label

The experiments use 8,416 Java classes from 25 releases of Camel, JEdit, Log4j, Lucene, POI, Synapse, and Velocity. Files are pinned to immutable Zenodo records with checksums in [`config/data_manifest.json`](../config/data_manifest.json); provenance, CC BY 4.0 terms, and release links are in [`docs/DATASET.md`](../docs/DATASET.md). One row represents one class in one release.

The source `bug` count is normalized as `bug_count`; the binary target is `defective = 1` when `bug_count > 0`, otherwise 0. Project, release, class name, bug count, and the derived target are never predictive features.

| Feature | Definition |
|---|---|
| `wmc` | Weighted methods per class |
| `dit` | Depth of inheritance tree |
| `noc` | Number of children |
| `cbo` | Coupling between object classes |
| `rfc` | Response for a class |
| `lcom` | Lack of cohesion in methods |
| `ca` | Afferent couplings |
| `ce` | Efferent couplings |
| `npm` | Number of public methods |
| `lcom3` | Normalized lack of cohesion in methods |
| `loc` | Lines of code |
| `dam` | Data access metric |
| `moa` | Measure of aggregation |
| `mfa` | Measure of functional abstraction |
| `cam` | Cohesion among methods of a class |
| `ic` | Inheritance coupling |
| `cbm` | Coupling between methods |
| `amc` | Average method complexity |
| `max_cc` | Maximum method cyclomatic complexity |
| `avg_cc` | Mean method cyclomatic complexity |

## Evaluation design

Every outer fold holds out all releases of one project and trains on the other projects. Pipelines are fitted fresh in each fold; imputation and scaling learn only from prediction-training rows. AP and top-20% recall are primary. Top-20% recall uses exactly `ceil(0.20 × target rows)` records, ranking by score descending and breaking ties by release, class name, then original order. Precision, recall, F1, and confusion matrices use the explicitly fixed threshold 0.5 unless a source-selected threshold is named.

Project-level results receive equal weight. Random three-project subsets are averaged within each target before averaging the seven targets; the 20 subsets are not treated as independent projects.

## Original baseline and source-validation threshold experiment

The fixed logistic model achieved macro AP **0.554**, precision **0.634**, recall **0.198**, and F1 **0.237** at threshold 0.5. Mean top-20% recall was **0.309**.

Thresholds selected only from source-project validation increased macro recall to **0.953** and F1 to **0.548**, while precision fell to **0.417**. Camel, JEdit, Lucene, POI, and Synapse flagged all or at least 95% of target records. Because the ranking scores did not change, AP was unchanged; the higher F1 describes a different operating cutoff, not better ranking.

## Fixed model comparison

All rows below are equal-weight means across the same seven outer projects. Configurations were declared before the run and were not selected from outer results.

| Model | AP | Top-20% recall | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| dummy_prior | 0.408 | 0.188 | 0.000 | 0.000 | 0.000 |
| logistic_regression | 0.554 | 0.309 | 0.634 | 0.198 | 0.237 |
| random_forest | 0.545 | 0.313 | 0.619 | 0.293 | 0.330 |
| hist_gradient_boosting | 0.514 | 0.286 | 0.548 | 0.300 | 0.329 |

Logistic regression had the highest macro AP. Random forest differed by **-0.010** AP and **+0.004** top-20% recall; histogram gradient boosting differed by **-0.040** AP. Random forest and boosting had higher F1 at 0.5, but that reflects different score distributions at a fixed cutoff and does not overturn the AP ranking result.

### Dummy ranking ties

Dummy prior assigns the same score to every target row. With the required deterministic identity tie-break, its observed macro top-20% recall was **0.188**. Under genuinely random selection of the same exact per-project budgets, the expected macro recall is the mean reviewed fraction, **0.200**. The observed value can be above or below that expectation because the deterministic release/class-name order happens to include a particular set of labels; it is not evidence that the dummy model ranks defects.

## Similar-project selection and exhaustive reference

Similarity preprocessing uses source-only median imputation and standard scaling. Each source and the unlabelled target is summarized by 20 feature medians plus 20 interquartile ranges; Euclidean distance selects the three closest sources. A new baseline logistic pipeline is then fitted only on the selected raw training rows. The similarity scaler is not reused for prediction.

| Training strategy | AP | Top-20% recall | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| all_sources | 0.554 | 0.309 | 0.634 | 0.198 | 0.237 |
| similar_sources | 0.534 | 0.299 | 0.574 | 0.310 | 0.315 |
| random_subset_mean | 0.547 | 0.308 | 0.584 | 0.300 | 0.302 |

Similarity selection changed AP by **-0.021** and top-20% recall by **-0.010** versus all sources. Versus the mean exhaustive three-source reference, the changes were **-0.013** AP and **-0.009** top-20% recall. It improved AP over all sources only for Log4j and regressed for the other six targets. Similarity therefore did not improve ranking overall.

Similarity increased macro F1 at 0.5, but behavior was heterogeneous: target-specific recall ranged from 0.027 to 0.871. Higher F1 at this cutoff does not establish improved ranking. The exhaustive reference matches the number of source projects, not training-row counts; unequal source-project sizes remain a confounder.

## Related work

- Jureczko and Madeyski introduced and analyzed the multi-project corpus and investigated project clustering for defect prediction: [*Towards Identifying Software Project Clusters with Regard to Defect Prediction*](https://doi.org/10.1145/1868328.1868342) (PROMISE 2010).
- Turhan, Menzies, Bener, and Di Stefano studied filtered cross-company data and within-company predictors: [*On the Relative Value of Cross-Company and Within-Company Data for Defect Prediction*](https://doi.org/10.1007/s10664-008-9103-7) (Empirical Software Engineering, 2009).
- Herbold evaluated training-data selection for cross-project prediction: [*Training Data Selection for Cross-Project Defect Prediction*](https://doi.org/10.1145/2499393.2499395) (PROMISE 2013).
- Nam, Pan, and Kim proposed transfer defect learning: [*Transfer Defect Learning*](https://doi.org/10.1109/ICSE.2013.6606584) (ICSE 2013).

These primary studies motivate cross-project transfer and source selection. Their datasets, selection rules, validation designs, and metrics differ, so their reported outcomes are not treated as directly comparable effect estimates for this experiment.

## Limitations

- Only seven historical Java projects are evaluated; results may not transfer to current languages, development practices, or organizations.
- Releases repeat classes over time, so rows are not necessarily temporally or structurally independent.
- Exact 20-metric vector matches occur across projects, sometimes with conflicting labels. A metric match alone does not prove copied code; no rows were removed automatically.
- Defect labels are historical observational data and may be incomplete or incorrectly linked.
- Projects have unequal sizes and defect prevalence. Three-project references do not guarantee equal training-row counts.
- Target metric distributions are available to similarity selection, so it is an unlabelled-target-access approach rather than a strictly source-only method.
- The fixed configurations are bounded comparisons, not an exhaustive model search; scores are not calibrated probabilities.
- No novelty, statistical-significance, causal, or production-readiness claim is established.

## Reproduction

From the repository root on PowerShell, after the one-time environment setup in the README:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.download
.\.venv\Scripts\python.exe -m defect_prediction.experiment
.\.venv\Scripts\python.exe -m defect_prediction.evaluate
.\.venv\Scripts\python.exe -m defect_prediction.lopo
.\.venv\Scripts\python.exe -m defect_prediction.source_validation
.\.venv\Scripts\python.exe -m defect_prediction.model_comparison
.\.venv\Scripts\python.exe -m defect_prediction.source_similarity
.\.venv\Scripts\python.exe -m defect_prediction.research_reporting
.\.venv\Scripts\python.exe -m defect_prediction.deployment
```

The correct similarity module is `defect_prediction.source_similarity`; there is no `source_similarity4` runner.

Configurations and protocols:

- [Original baseline configuration](../config/experiment.json)
- [Fixed model-comparison configuration](../config/model_comparison.json) and [protocol](../docs/MODEL_COMPARISON_PROTOCOL.md)
- [Similarity-selection configuration](../config/source_similarity.json) and [protocol](../docs/SOURCE_SIMILARITY_PROTOCOL.md)
- [Dataset manifest](../config/data_manifest.json) and [dataset documentation](../docs/DATASET.md)

Research models and reports remain historical. The API continues to serve the separately packaged, checksum-pinned `logistic-regression-jedit-holdout-v1` artifact.
