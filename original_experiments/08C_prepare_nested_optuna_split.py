from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 13A
# NESTED PROJECT SPLIT FOR OPTUNA
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

CROSS_PROJECT_DIR = (
    BASE_DIR
    / "results_cross_project"
)

TRAIN_FILE = (
    CROSS_PROJECT_DIR
    / "cross_project_train.csv"
)

OFFICIAL_VAL_FILE = (
    CROSS_PROJECT_DIR
    / "cross_project_validation.csv"
)

OFFICIAL_TEST_FILE = (
    CROSS_PROJECT_DIR
    / "cross_project_test.csv"
)

OUTPUT_DIR = (
    CROSS_PROJECT_DIR
    / "nested_optuna"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INNER_TRAIN_FILE = (
    OUTPUT_DIR
    / "inner_train.csv"
)

INNER_OPTUNA_VAL_FILE = (
    OUTPUT_DIR
    / "inner_optuna_validation.csv"
)

print("=" * 90)
print("STEP 13A - NESTED PROJECT SPLIT FOR OPTUNA")
print("=" * 90)

# ============================================================
# LOAD EXISTING CROSS-PROJECT SPLITS
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

official_val_df = pd.read_csv(
    OFFICIAL_VAL_FILE,
    low_memory=False
)

official_test_df = pd.read_csv(
    OFFICIAL_TEST_FILE,
    low_memory=False
)

# ============================================================
# PROJECT LISTS
# ============================================================

development_projects = sorted(
    train_df[
        "project_name"
    ]
    .unique()
    .tolist()
)

official_val_projects = set(
    official_val_df[
        "project_name"
    ]
    .unique()
)

official_test_projects = set(
    official_test_df[
        "project_name"
    ]
    .unique()
)

print(
    "\nDevelopment projects:",
    len(
        development_projects
    )
)

print(
    "Official validation projects:",
    len(
        official_val_projects
    )
)

print(
    "Official test projects:",
    len(
        official_test_projects
    )
)

# ============================================================
# SPLIT 14 DEVELOPMENT PROJECTS
#
# 11 projects = inner model training
# 3 projects  = Optuna hyperparameter evaluation
# ============================================================

inner_train_projects, \
inner_optuna_projects = train_test_split(

    development_projects,

    test_size=3,

    random_state=SEED
)

inner_train_projects = set(
    inner_train_projects
)

inner_optuna_projects = set(
    inner_optuna_projects
)

# ============================================================
# CREATE DATASETS
# ============================================================

inner_train_df = train_df[
    train_df[
        "project_name"
    ].isin(
        inner_train_projects
    )
].copy()

inner_optuna_df = train_df[
    train_df[
        "project_name"
    ].isin(
        inner_optuna_projects
    )
].copy()

# ============================================================
# VERIFY ALL PROJECT SEPARATION
# ============================================================

assert len(
    inner_train_projects
    & inner_optuna_projects
) == 0

assert len(
    inner_train_projects
    & official_val_projects
) == 0

assert len(
    inner_train_projects
    & official_test_projects
) == 0

assert len(
    inner_optuna_projects
    & official_val_projects
) == 0

assert len(
    inner_optuna_projects
    & official_test_projects
) == 0

assert len(
    official_val_projects
    & official_test_projects
) == 0

print(
    "\nAll overlap checks: PASSED"
)

# ============================================================
# SUMMARY FUNCTION
# ============================================================

def describe(
    name,
    dataframe
):

    rows = len(
        dataframe
    )

    projects = (
        dataframe[
            "project_name"
        ]
        .nunique()
    )

    failed = int(
        dataframe[
            "label"
        ]
        .sum()
    )

    successful = (
        rows
        - failed
    )

    failure_rate = (
        failed
        / rows
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        name
    )

    print(
        "=" * 90
    )

    print(
        "Rows:",
        rows
    )

    print(
        "Projects:",
        projects
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Failure rate:",
        round(
            failure_rate,
            4
        )
    )

    print(
        "\nProject list:"
    )

    for project in sorted(
        dataframe[
            "project_name"
        ]
        .unique()
    ):

        print(
            project
        )

# ============================================================
# DISPLAY
# ============================================================

describe(
    "INNER TRAINING PROJECTS",
    inner_train_df
)

describe(
    "INNER OPTUNA VALIDATION PROJECTS",
    inner_optuna_df
)

describe(
    "OFFICIAL VALIDATION PROJECTS - UNTOUCHED BY OPTUNA",
    official_val_df
)

describe(
    "OFFICIAL TEST PROJECTS - COMPLETELY UNTOUCHED",
    official_test_df
)

# ============================================================
# SAVE INNER SPLITS
# ============================================================

inner_train_df.to_csv(
    INNER_TRAIN_FILE,
    index=False
)

inner_optuna_df.to_csv(
    INNER_OPTUNA_VAL_FILE,
    index=False
)

# ============================================================
# PROJECT ASSIGNMENT REPORT
# ============================================================

records = []

for project in sorted(
    inner_train_projects
):

    records.append(
        {
            "project_name":
                project,

            "role":
                "Inner Train"
        }
    )

for project in sorted(
    inner_optuna_projects
):

    records.append(
        {
            "project_name":
                project,

            "role":
                "Inner Optuna Validation"
        }
    )

for project in sorted(
    official_val_projects
):

    records.append(
        {
            "project_name":
                project,

            "role":
                "Official Validation"
        }
    )

for project in sorted(
    official_test_projects
):

    records.append(
        {
            "project_name":
                project,

            "role":
                "Official Test"
        }
    )

assignment_df = pd.DataFrame(
    records
)

assignment_df.to_csv(
    OUTPUT_DIR
    / "08C_nested_project_assignments.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "NESTED OPTUNA SPLIT SUMMARY"
)

print(
    "=" * 90
)

print(
    "Inner training projects:",
    len(
        inner_train_projects
    )
)

print(
    "Inner Optuna-validation projects:",
    len(
        inner_optuna_projects
    )
)

print(
    "Official validation projects:",
    len(
        official_val_projects
    )
)

print(
    "Official test projects:",
    len(
        official_test_projects
    )
)

print(
    "\nTotal distinct projects:",
    len(
        inner_train_projects
        | inner_optuna_projects
        | official_val_projects
        | official_test_projects
    )
)

print(
    "\nOptuna will NOT see official validation projects."
)

print(
    "Optuna will NEVER see official test projects."
)

print(
    "\nSaved:"
)

print(
    INNER_TRAIN_FILE
)

print(
    INNER_OPTUNA_VAL_FILE
)

print(
    "\nSTEP 13A COMPLETED SUCCESSFULLY"
)