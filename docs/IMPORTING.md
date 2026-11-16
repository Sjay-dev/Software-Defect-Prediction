# Compatible prediction-file imports

The prediction dashboard imports files locally and sends only the deployed model's 20 numeric feature fields plus a stable `record_id` to `POST /predict`. Importing a file never trains, tunes, calibrates, or otherwise modifies the saved model.

## Implemented formats

| Format | Implemented behavior | Deliberate restrictions |
|---|---|---|
| CSV | Comma, semicolon, or tab delimiter is detected; the user can override it. UTF-8 BOMs, surrounding header whitespace, and header case differences are normalized for matching. | One header row; no nested data. |
| TSV | Parsed as tab-delimited by default, with the same validation as CSV. | One header row. |
| JSON | One record object or an array of record objects. The union of scalar properties is previewed as columns. | Nested objects and arrays are rejected as unsupported cell values. |
| XLSX | The first row is the header. Every worksheet is listed so the user can select the intended sheet. | `.xls` and `.xlsm` are not accepted. Formula cells, including cells with cached results, are rejected rather than evaluated. |
| ARFF | Dense ARFF with `numeric`, `real`, `integer`, `string`, or nominal `{...}` attributes. Quoted attribute names and `?` missing values are supported. | Sparse rows, relational attributes, date attributes, directives after `@data`, and multiline values are rejected with an explanation. |

Downloadable CSV, TSV, JSON, XLSX, and ARFF templates are available from the Prediction screen.

## Recognised layouts and exact mappings

### Dashboard schema

The preferred layout is:

```text
record_id,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc
```

The metric names refer to the exact definitions documented in [DATASET.md](DATASET.md). Matching ignores header case, a leading UTF-8 BOM, and surrounding whitespace; it does not use fuzzy matching or substitute similarly named metrics.

Optional metadata columns explicitly recognised in this layout are `project`, `release` or `version`, and `class_name`. `bug_count` is recognised as a nonnegative count and `defective` as a documented binary indicator. These outcome columns remain separate from predictor inputs.

### Original Jureczko/PROMISE layout

The downloaded corpus files, including the original Ant-style CSV, deliberately contain two `name` headers:

```text
name,version,name,<20 metrics>,bug
```

This exact recognised layout maps:

| Source column | Role |
|---|---|
| first `name` | project |
| `version` | release |
| second `name` | original fully qualified class name |
| the canonical 20 metric columns | model inputs, in the deployed feature order |
| `bug` | documented defect count; excluded from model inputs |

This is the only supported duplicate-header exception. Any other repeated header, including duplicates that differ only by case or whitespace, is rejected before rows are converted to records.

### Unfamiliar layouts

The preview shows every recognised metric and every missing mapping. A user may explicitly map a source column to each missing required metric. No similarly named metric is selected automatically, and one source column cannot satisfy multiple required metrics. Ambiguous columns such as `label`, `target`, `class`, or a generic `bug` outside the recognised Jureczko layout require the user to declare their meaning or mark them as excluded.

Files missing one or more required metric definitions cannot be scored. An arbitrary business CSV is therefore not compatible merely because it has 20 numeric columns.

## Identifiers, class names, and outcomes

- A valid, unique source `record_id` is preserved exactly.
- Blank, overlong, or duplicated source IDs are replaced deterministically. Project/release/class metadata is preferred; otherwise the file name and source row are used.
- Metadata collisions receive deterministic row suffixes. API IDs are capped at 200 characters with a stable hash suffix when necessary.
- Source row numbers and original class names are retained for preview and result export.
- Established labels are preserved for optional joined export and optional descriptive evaluation. They never enter model inputs.
- `bug_count` and the Jureczko `bug` field become a binary recorded outcome only through the documented rule `count > 0`.
- A `defective` indicator accepts only `0`, `1`, `false`, or `true`. Uninterpretable label values remain exportable but are excluded from evaluation.
- Unlabelled files are fully supported.

If an imported labelled file contains only one outcome class, confusion counts and any defined threshold metrics are still displayed. Average precision and ROC-AUC are reported as undefined, with an explanation; no artificial second class is created.

## Validation and partial imports

The preview reports the detected format, selected delimiter or worksheet, layout, metric mapping count, identifier strategy, outcome columns, excluded columns, and row-level validation findings. Missing metric cells remain `null` and are handled only by the preprocessing bundled in the saved inference pipeline. Original finite numeric values are not clipped, transformed, or imputed by the importer.

Nonnumeric values, `NaN`, infinity, booleans used as metrics, formulas, and unsupported cell structures are invalid. Invalid records are never silently discarded. When at least one valid record remains, prediction stays disabled until the user explicitly checks **Use only the valid rows**; the excluded source rows and columns are shown. If every record is invalid, the file cannot be scored.

## Limits, batching, and safety

- Maximum compressed/source file size: 10 MiB.
- Maximum parsed records: 10,000.
- Maximum source columns or ARFF attributes: 256.
- API batch size: the running service's `max_batch_size` (currently 100, with an absolute backend maximum of 100).
- Retry policy: one bounded retry per failed dashboard batch.
- Cancellation: stops further batches and displays no partial result table.

The dashboard pins the model version reported when the import begins and rejects a response from another version. Global ranks are computed across all records only after every batch succeeds. A failed or cancelled run is explicitly incomplete and is never presented as full-file evaluation.

XLSX formulas and cached formula results are not used, macros are not supported, ARFF is parsed as data without evaluation or object construction, and no uploaded content is executed.

## Result exports

The results CSV contains stable `record_id`, source row (when available), original class name (when available), model score, global `rank`, and model version. If established outcomes exist, the user can include or omit them from the same export. Outcome columns are prefixed with `outcome_`.

## Scope boundary

Raw Java source is not a metric table. Supporting it requires a separately validated static-metric extractor that produces these exact 20 definitions. A dataset with different feature definitions requires a separately trained and evaluated model. Neither is simulated by renaming unrelated columns.

## Implemented and tested matrix

Automated frontend tests cover:

- comma, semicolon, and tab CSV plus TSV-compatible parsing;
- single-record and array JSON;
- XLSX worksheet selection and formula rejection;
- dense ARFF plus sparse/unsupported-structure rejection;
- UTF-8 BOM, whitespace, and case-normalised headers;
- the original two-`name` Jureczko/PROMISE layout and other duplicate headers;
- unfamiliar explicit mappings and missing metrics;
- missing, nonnumeric, and nonfinite values with row/column reporting;
- identifier collisions, label exclusion, ambiguous outcome mapping state, and one-class evaluation;
- bounded batching, retry, cancellation, one-version enforcement, and global ranking; and
- identical model inputs and scores for the same record imported through CSV, JSON, ARFF, and XLSX.

Run the verification with:

```powershell
npm --prefix frontend test
npm --prefix frontend run build
```
