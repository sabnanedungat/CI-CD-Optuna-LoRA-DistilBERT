import re
import pandas as pd

INPUT_FILE = "travis_sample.csv"
OUTPUT_FILE = "cicd_logs_leakage_safe.csv"

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("STEP 1 - LEAKAGE-SAFE DATASET PREPARATION")
print("=" * 60)
print("Original shape:", df.shape)

# ------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------
keep_cols = [
    "project_name",
    "lang",
    "branch",
    "status",
    "duration",
    "frameworks",
    "tests_ran",
    "src_churn",
    "test_churn",
    "files_added",
    "files_deleted",
    "files_modified",
    "num_jobs"
]

missing_cols = [col for col in keep_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        "Missing columns: " + ", ".join(missing_cols)
    )

df = df[keep_cols].copy()

# ------------------------------------------------------------
# Create target label
# ------------------------------------------------------------
df = df.dropna(subset=["status"]).copy()

df["status"] = (
    df["status"]
    .astype(str)
    .str.lower()
    .str.strip()
)

success_values = [
    "passed",
    "success",
    "fixed"
]

failure_values = [
    "failed",
    "errored"
]

df = df[
    df["status"].isin(
        success_values + failure_values
    )
].copy()

df["label"] = df["status"].apply(
    lambda x: 1 if x in failure_values else 0
)

# ------------------------------------------------------------
# Fill missing values
# ------------------------------------------------------------
for col in keep_cols:
    if col != "status":
        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
        )

# ------------------------------------------------------------
# Build leakage-safe log_text
#
# IMPORTANT:
# failed_tests and tests_failed are NOT used.
# status is also NOT included in log_text.
# ------------------------------------------------------------
df["log_text"] = (
    "project " + df["project_name"] +
    " language " + df["lang"] +
    " branch " + df["branch"] +
    " duration " + df["duration"] +
    " frameworks " + df["frameworks"] +
    " tests_ran " + df["tests_ran"] +
    " src_churn " + df["src_churn"] +
    " test_churn " + df["test_churn"] +
    " files_added " + df["files_added"] +
    " files_deleted " + df["files_deleted"] +
    " files_modified " + df["files_modified"] +
    " num_jobs " + df["num_jobs"]
)

# ------------------------------------------------------------
# Remove direct outcome-revealing terms
# ------------------------------------------------------------
LEAKAGE_TERMS = [
    "pass",
    "passed",
    "fail",
    "failed",
    "failure",
    "success",
    "successful",
    "error",
    "errored",
    "status",
    "result",
    "label",
    "build_status",
    "build_state"
]

pattern = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(term)
        for term in LEAKAGE_TERMS
    )
    + r")\b",
    flags=re.IGNORECASE
)


def clean_log_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove direct leakage terms
    text = pattern.sub(
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["log_text"] = df["log_text"].apply(
    clean_log_text
)

# ------------------------------------------------------------
# Keep project_name
# This is required later for cross-project validation
# ------------------------------------------------------------
final_df = df[
    [
        "project_name",
        "log_text",
        "label"
    ]
].copy()

final_df = final_df.dropna()

final_df = final_df[
    final_df["log_text"].str.strip() != ""
].copy()

# ------------------------------------------------------------
# Remove exact duplicate rows
# ------------------------------------------------------------
before_duplicates = len(final_df)

final_df = (
    final_df
    .drop_duplicates()
    .reset_index(drop=True)
)

duplicates_removed = (
    before_duplicates - len(final_df)
)

# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print(
    "Final shape:",
    final_df.shape
)

print("\nLabel distribution:")

print(
    final_df["label"]
    .value_counts()
    .sort_index()
)

successful = int(
    (final_df["label"] == 0).sum()
)

failed = int(
    (final_df["label"] == 1).sum()
)

failure_ratio = float(
    final_df["label"].mean()
)

print(
    "\nSuccessful builds:",
    successful
)

print(
    "Failed builds:",
    failed
)

print(
    "Failure ratio:",
    round(failure_ratio, 4)
)

print(
    "\nUnique projects:",
    final_df["project_name"].nunique()
)

print(
    "Duplicate rows removed:",
    duplicates_removed
)

average_words = (
    final_df["log_text"]
    .apply(
        lambda x: len(
            str(x).split()
        )
    )
    .mean()
)

print(
    "Average words per log:",
    round(average_words, 2)
)

# ------------------------------------------------------------
# Save new dataset
# ------------------------------------------------------------
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "\nKeep your previous dataset unchanged."
)

print(
    "Use this new dataset for the strengthened JSS experiments."
)