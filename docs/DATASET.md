# Dataset provenance and meaning

This milestone uses seven projects from the Jureczko software-defect corpus in the SeaCraft community (formerly the PROMISE repository). Each Zenodo record is immutable and its exact CSV files and MD5 checksums are pinned in `config/data_manifest.json`.

## Provenance and usage terms

- Creator recorded by these Zenodo records: Marian Jureckzo (the record's spelling). The cited publication and other scholarly sources spell the author's surname Marian Jureczko.
- Collection/publication reference: M. Jureczko and L. Madeyski, *Towards Identifying Software Project Clusters with Regard to Defect Prediction*, PROMISE 2010, DOI [10.1145/1868328.1868342](https://doi.org/10.1145/1868328.1868342).
- Repository records: [Camel](https://zenodo.org/records/268442), [JEdit](https://zenodo.org/records/268458), [Log4j](https://zenodo.org/records/268451), [Lucene](https://zenodo.org/records/322458), [POI](https://zenodo.org/records/322443), [Synapse](https://zenodo.org/records/322449), and [Velocity](https://zenodo.org/records/322455).
- Zenodo API metadata reports `cc-by-4.0` for each selected record. CC BY 4.0 permits sharing and adapting the data with attribution and a link to the license. See the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
- The raw data is downloaded from Zenodo rather than copied into this repository. Generated work must retain the attribution above.

## Unit of observation and label

One row describes one Java class in one project release. The source `bug` field is the number of known defects associated with that class. This project preserves it as `bug_count` and derives a binary classification target:

```text
defective = 1 if bug_count > 0
defective = 0 if bug_count == 0
```

`bug_count` and `defective` are outcomes, so neither is allowed into the predictive features. `project`, `release`, and `class_name` identify a row and are also excluded from the features. This prevents the model from memorizing project or class names.

## Compatible feature set

Every selected release has the same 20 class-level static-code metrics:

| Feature | Plain-language meaning |
|---|---|
| `wmc` | Weighted methods per class |
| `dit` | Depth of the inheritance tree |
| `noc` | Number of child classes |
| `cbo` | Coupling between object classes |
| `rfc` | Response for a class |
| `lcom` | Lack of cohesion in methods |
| `ca` | Afferent (incoming) couplings |
| `ce` | Efferent (outgoing) couplings |
| `npm` | Number of public methods |
| `lcom3` | Normalized lack of cohesion |
| `loc` | Lines of code |
| `dam` | Data access metric |
| `moa` | Measure of aggregation |
| `mfa` | Measure of functional abstraction |
| `cam` | Cohesion among class methods |
| `ic` | Inheritance coupling |
| `cbm` | Coupling between methods |
| `amc` | Average method complexity/size |
| `max_cc` | Maximum method cyclomatic complexity |
| `avg_cc` | Mean method cyclomatic complexity |

The names and definitions are corroborated by the corpus documentation and later data descriptions, including the feature table in [Process metrics for software defect prediction in object-oriented programs](https://doi.org/10.1049/iet-sen.2018.5439).

## Known limitations

These are historical observational labels, not perfect ground truth. Defects may be missed or linked to the wrong class, repeated classes occur across releases, projects have different defect rates, and the collection is older Java software. The experiment therefore reports a held-out-project baseline, not a claim that the model is deployment-ready.
