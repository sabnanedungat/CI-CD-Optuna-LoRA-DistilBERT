from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# ============================================================
# STEP 11
# CROSS-PROJECT TRAIN / VALIDATION / TEST SPLIT
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "cicd_multiproject_leakage_safe.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results_cross_project"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "cross_project_train.csv"
)

VAL_FILE = (
    OUTPUT_DIR
    / "cross_project_validation.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "cross_project_test.csv"
)

print("=" * 85)
print("STEP 11 - CROSS-PROJECT SPLIT PREPARATION")
print("=" * 85)

# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

required_columns = [
    "project_name",
    "log_text",
    "label"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )

df = df.dropna(
    subset=required_columns
).copy()

df["project_name"] = (
    df["project_name"]
    .astype(str)
)

df["log_text"] = (
    df["log_text"]
    .astype(str)
)

df["label"] = (
    df["label"]
    .astype(int)
)

print(
    "\nTotal rows:",
    len(df)
)

print(
    "Total projects:",
    df["project_name"].nunique()
)

print(
    "\nOverall label distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)

# ============================================================
# PROJECT-LEVEL SUMMARY
# ============================================================

project_summary = (

    df
    .groupby(
        "project_name"
    )
    .agg(
        total_builds=(
            "label",
            "size"
        ),

        failed_builds=(
            "label",
            "sum"
        )
    )
    .reset_index()
)

project_summary[
    "successful_builds"
] = (

    project_summary[
        "total_builds"
    ]

    - project_summary[
        "failed_builds"
    ]
)

project_summary[
    "failure_rate"
] = (

    project_summary[
        "failed_builds"
    ]

    / project_summary[
        "total_builds"
    ]
)

project_summary = (

    project_summary
    .sort_values(
        "project_name"
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nProject summary:"
)

print(
    project_summary.to_string(
        index=False
    )
)

# ============================================================
# PROJECT-LEVEL SPLIT
#
# 20 projects:
#
# 14 projects training
# 3 projects validation
# 3 projects testing
#
# No project overlap is permitted.
# ============================================================

projects = (
    project_summary[
        "project_name"
    ]
    .to_numpy()
)

train_projects, temp_projects = train_test_split(

    projects,

    test_size=0.30,

    random_state=SEED
)

val_projects, test_projects = train_test_split(

    temp_projects,

    test_size=0.50,

    random_state=SEED
)

train_projects = sorted(
    train_projects.tolist()
)

val_projects = sorted(
    val_projects.tolist()
)

test_projects = sorted(
    test_projects.tolist()
)

# ============================================================
# CREATE ROW-LEVEL DATASETS
# ============================================================

train_df = df[
    df[
        "project_name"
    ].isin(
        train_projects
    )
].copy()

val_df = df[
    df[
        "project_name"
    ].isin(
        val_projects
    )
].copy()

test_df = df[
    df[
        "project_name"
    ].isin(
        test_projects
    )
].copy()

# ============================================================
# VERIFY NO PROJECT LEAKAGE
# ============================================================

train_set = set(
    train_projects
)

val_set = set(
    val_projects
)

test_set = set(
    test_projects
)

train_val_overlap = (
    train_set
    & val_set
)

train_test_overlap = (
    train_set
    & test_set
)

val_test_overlap = (
    val_set
    & test_set
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "Project leakage detected."
    )

print(
    "\nProject overlap check: PASSED"
)

# ============================================================
# SPLIT SUMMARY FUNCTION
# ============================================================

def describe_split(
    name,
    dataframe
):

    total = len(
        dataframe
    )

    failures = int(
        dataframe[
            "label"
        ].sum()
    )

    successes = (
        total
        - failures
    )

    failure_rate = (

        failures
        / total

        if total > 0

        else 0
    )

    print(
        "\n"
        + "=" * 85
    )

    print(
        name
    )

    print(
        "=" * 85
    )

    print(
        "Rows:",
        total
    )

    print(
        "Projects:",
        dataframe[
            "project_name"
        ].nunique()
    )

    print(
        "Successful:",
        successes
    )

    print(
        "Failed:",
        failures
    )

    print(
        "Failure rate:",
        round(
            failure_rate,
            4
        )
    )

    print(
        "\nProjects:"
    )

    print(
        "\n".join(
            sorted(
                dataframe[
                    "project_name"
                ].unique()
            )
        )
    )

    return {
        "split":
            name,

        "rows":
            total,

        "projects":
            dataframe[
                "project_name"
            ].nunique(),

        "successful":
            successes,

        "failed":
            failures,

        "failure_rate":
            failure_rate
    }

# ============================================================
# DISPLAY SPLITS
# ============================================================

summary_rows = []

summary_rows.append(
    describe_split(
        "TRAINING PROJECTS",
        train_df
    )
)

summary_rows.append(
    describe_split(
        "VALIDATION PROJECTS",
        val_df
    )
)

summary_rows.append(
    describe_split(
        "TEST PROJECTS",
        test_df
    )
)

# ============================================================
# PROJECT-BY-LABEL DISTRIBUTION
# ============================================================

combined_project_report = (

    df
    .groupby(
        "project_name"
    )
    .agg(
        Rows=(
            "label",
            "size"
        ),

        Failed=(
            "label",
            "sum"
        )
    )
    .reset_index()
)

combined_project_report[
    "Successful"
] = (

    combined_project_report[
        "Rows"
    ]

    - combined_project_report[
        "Failed"
    ]
)

combined_project_report[
    "Failure_Rate"
] = (

    combined_project_report[
        "Failed"
    ]

    / combined_project_report[
        "Rows"
    ]
)

# ============================================================
# ADD SPLIT LABEL
# ============================================================

def get_split(
    project
):

    if project in train_set:

        return "Train"

    if project in val_set:

        return "Validation"

    if project in test_set:

        return "Test"

    return "Unknown"


combined_project_report[
    "Split"
] = (

    combined_project_report[
        "project_name"
    ]
    .apply(
        get_split
    )
)

combined_project_report = (

    combined_project_report
    .sort_values(
        [
            "Split",
            "project_name"
        ]
    )
)

# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

pd.DataFrame(
    summary_rows
).to_csv(

    OUTPUT_DIR
    / "08A_cross_project_split_summary.csv",

    index=False
)

combined_project_report.to_csv(

    OUTPUT_DIR
    / "08A_project_assignment_report.csv",

    index=False
)

# ============================================================
# FINAL VERIFICATION
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "FINAL CROSS-PROJECT SPLIT"
)

print(
    "=" * 85
)

print(
    "Training projects:",
    len(
        train_projects
    )
)

print(
    "Validation projects:",
    len(
        val_projects
    )
)

print(
    "Test projects:",
    len(
        test_projects
    )
)

print(
    "\nTrain/Validation overlap:",
    len(
        train_val_overlap
    )
)

print(
    "Train/Test overlap:",
    len(
        train_test_overlap
    )
)

print(
    "Validation/Test overlap:",
    len(
        val_test_overlap
    )
)

print(
    "\nSaved:"
)

print(
    TRAIN_FILE
)

print(
    VAL_FILE
)

print(
    TEST_FILE
)

print(
    "\nSTEP 11 COMPLETED SUCCESSFULLY"
)