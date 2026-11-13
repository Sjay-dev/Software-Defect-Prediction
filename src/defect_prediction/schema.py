"""The fixed, documented schema shared by loading, training, and evaluation."""

FEATURE_COLUMNS = (
    "wmc",
    "dit",
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "max_cc",
    "avg_cc",
)

METADATA_COLUMNS = ("project", "release", "class_name")
LABEL_SOURCE_COLUMN = "bug_count"
TARGET_COLUMN = "defective"

SOURCE_HEADER = (
    "name",
    "version",
    "name",
    *FEATURE_COLUMNS,
    "bug",
)

NORMALIZED_COLUMNS = (
    *METADATA_COLUMNS,
    *FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    TARGET_COLUMN,
)

FEATURE_DEFINITIONS = {
    "wmc": "Weighted methods per class",
    "dit": "Depth of inheritance tree",
    "noc": "Number of children",
    "cbo": "Coupling between object classes",
    "rfc": "Response for a class",
    "lcom": "Lack of cohesion in methods",
    "ca": "Afferent couplings",
    "ce": "Efferent couplings",
    "npm": "Number of public methods",
    "lcom3": "Normalized lack of cohesion in methods",
    "loc": "Lines of code",
    "dam": "Data access metric",
    "moa": "Measure of aggregation",
    "mfa": "Measure of functional abstraction",
    "cam": "Cohesion among methods of a class",
    "ic": "Inheritance coupling",
    "cbm": "Coupling between methods",
    "amc": "Average method complexity",
    "max_cc": "Maximum method cyclomatic complexity",
    "avg_cc": "Mean method cyclomatic complexity",
}

