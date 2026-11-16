"""Compare three classical baselines on the *validation* split only.

For the candidate 5,710-build TravisTorrent dataset.
This controlled ablation REMOVES language because historical language timing is uncertain.
- Uses identical records and shared train-fitted feature preprocessing.
- Evaluates Logistic Regression, Random Forest, and (if installed) XGBoost.
- Chooses each model's F1 threshold using validation data, so validation F1
  is an *optimistic model-selection score*, NOT a final journal result.
- NEVER scores the test set, changes split files, or overwrites results.
- Feature availability before build start is NOT fully verified.

Run in the CICD_JSS_Strengthened folder:
    python 01L_no_language_three_baselines_VALIDATION.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "travis_unique_builds_20projects_CANDIDATE.csv"
MANIFEST_FILE = ROOT / "unique_builds_chronological_split_manifest.csv"
RANDOM_STATE = 42

if not SAMPLE_FILE.is_file() or not MANIFEST_FILE.is_file():
    raise FileNotFoundError(
        "Put the script alongside travis_unique_builds_20projects_CANDIDATE.csv "
        "and unique_builds_chronological_split_manifest.csv"
    )

sample = pd.read_csv(SAMPLE_FILE, dtype={"build_id": "string"}, low_memory=False).reset_index(drop=True)
manifest = pd.read_csv(MANIFEST_FILE, dtype={"build_id": "string"}, low_memory=False)
required_manifest = {"sample_row_id", "project_name", "build_id", "label", "split"}
if not required_manifest.issubset(manifest.columns):
    raise ValueError(f"Missing manifest columns: {sorted(required_manifest - set(manifest.columns))}")
if len(sample) != len(manifest) or len(sample) != 5710:
    raise ValueError("Sample/manifest row counts differ or are not 5710")
if manifest["sample_row_id"].duplicated().any():
    raise ValueError("Repeated sample_row_id in manifest")
manifest = manifest.set_index("sample_row_id").reindex(range(len(sample)))
if manifest[["project_name", "build_id", "label", "split"]].isna().any().any():
    raise ValueError("Missing or misaligned manifest records")
for col in ("project_name", "build_id"):
    if not sample[col].astype(str).equals(manifest[col].astype(str)):
        raise ValueError(f"Sample/manifest mismatch: {col}")
if sample.duplicated(["project_name", "build_id"]).any():
    raise ValueError("Duplicate CI builds detected")

status = sample["status"].astype(str).str.strip().str.lower()
if not status.isin({"passed", "success", "fixed", "failed", "errored"}).all():
    raise ValueError("Unexpected build status")
y = status.isin(["failed", "errored"]).astype(int).reset_index(drop=True)
if not np.array_equal(y.to_numpy(), manifest["label"].astype(int).to_numpy()):
    raise ValueError("Sample/manifest labels do not match")

split = manifest["split"].astype(str)
if not split.isin({"train", "validation", "test"}).all():
    raise ValueError("Unexpected split value")

# Safety check: no project/build identifier crosses partitions.
groups = pd.DataFrame({
    "key": sample["project_name"].astype(str) + "|" + sample["build_id"].astype(str),
    "split": split,
})
if groups.groupby("key")["split"].nunique().max() != 1:
    raise ValueError("CI build present in more than one partition")

# Check temporal ordering within each project.
dates = pd.to_datetime(sample["started_at"], errors="coerce", utc=True)
if dates.isna().any():
    raise ValueError("Missing/invalid started_at timestamp")
for project, index in sample.groupby("project_name").groups.items():
    ix = list(index)
    for earlier, later in (("train", "validation"), ("validation", "test")):
        early_times = dates.iloc[ix][split.iloc[ix] == earlier]
        late_times = dates.iloc[ix][split.iloc[ix] == later]
        if len(early_times) and len(late_times) and early_times.max() > late_times.min():
            raise ValueError(f"Within-project chronology violation: {project}")

# IMPORTANT: these four columns are provisional pre-build candidates.
X = pd.DataFrame({
    "branch": sample["branch"].fillna("unknown").astype(str),
    "is_pr": sample["is_pr"].fillna("unknown").astype(str),
    "num_commits": pd.to_numeric(sample["num_commits"], errors="coerce"),
})
train = split.eq("train").to_numpy()
validation = split.eq("validation").to_numpy()
test = split.eq("test").to_numpy()
if not train.any() or not validation.any() or not test.any():
    raise ValueError("An expected partition is empty")

# Train-only fitted preprocessing shared by all models.
preprocessor = ColumnTransformer([
    ("cats", OneHotEncoder(handle_unknown="ignore"), ["branch", "is_pr"]),
    ("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), ["num_commits"]),
])
X_train = preprocessor.fit_transform(X.loc[train])
X_val = preprocessor.transform(X.loc[validation])
y_train = y.loc[train].to_numpy()
y_val = y.loc[validation].to_numpy()

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}
try:
    from xgboost import XGBClassifier
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=neg / max(pos, 1),
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost is not installed. To add it, run: python -m pip install xgboost")

print("=" * 72)
print("NO-LANGUAGE ABLATION — UNIQUE-BUILD VALIDATION ONLY")
print("=" * 72)
print("Features: branch, is_pr, num_commits (language removed; timing still provisional)")
print("Training rows:", int(train.sum()))
print("Validation rows:", int(validation.sum()))
print("Test rows NOT evaluated:", int(test.sum()))
print("Validation failure fraction:", f"{y_val.mean():.4f}")
print("Note: validation-optimized F1 is not an unbiased final test estimate.")

results = []
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    options = [(f1_score(y_val, scores >= t, zero_division=0), float(t))
               for t in thresholds]
    best_f1, threshold = max(options, key=lambda item: (item[0], -item[1]))
    predicted = (scores >= threshold).astype(int)
    results.append({
        "model": name,
        "val_threshold": round(threshold, 2),
        "val_precision": round(precision_score(y_val, predicted, zero_division=0), 4),
        "val_recall": round(recall_score(y_val, predicted, zero_division=0), 4),
        "val_f1_selected": round(best_f1, 4),
        "val_ap": round(average_precision_score(y_val, scores), 4),
        "val_roc_auc": round(roc_auc_score(y_val, scores), 4),
    })

print("\n" + "=" * 72)
print("VALIDATION COMPARISON (NOT PUBLICATION RESULTS)")
print("=" * 72)
print(pd.DataFrame(results).to_string(index=False))
print("\nNo test predictions generated and no files changed.")
print("CAUTION: Establish pre-build feature availability before drawing claims.")

