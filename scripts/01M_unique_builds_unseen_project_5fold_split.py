"""Create 5 project-disjoint train/validation/test folds for the unique-build sample.

Each project's builds belong wholly to one partition WITHIN each fold.
Each project is held out as TEST in exactly one fold. This is NOT a
chronological split and must not be confused with the existing temporal test.
No models are trained, no test metrics are computed, and no existing files
are overwritten. Requires pandas and scikit-learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "travis_unique_builds_20projects_CANDIDATE.csv"
OUTPUT_PATH = ROOT / "unique_builds_unseen_project_5fold_manifest.csv"

if not SAMPLE_PATH.is_file():
    raise FileNotFoundError(f"Missing dataset: {SAMPLE_PATH}")
if OUTPUT_PATH.exists():
    raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT_PATH}")

sample = pd.read_csv(SAMPLE_PATH, dtype={"build_id": "string"}, low_memory=False)
required = {"project_name", "build_id", "status"}
if not required.issubset(sample.columns):
    raise ValueError(f"Missing columns: {required - set(sample.columns)}")
if len(sample) != 5710 or sample["project_name"].nunique() != 20:
    raise ValueError("Expected exactly 5,710 observations from 20 projects")
if sample[["project_name", "build_id"]].isna().any().any():
    raise ValueError("Project or build ID is missing")
if sample.duplicated(["project_name", "build_id"]).any():
    raise ValueError("Repeated project/build IDs in unique-build dataset")

status = sample["status"].astype(str).str.strip().str.lower()
if not status.isin(["passed", "success", "fixed", "failed", "errored"]).all():
    raise ValueError("Unexpected status value")
y = status.isin(["failed", "errored"]).astype(int).to_numpy()
if "label" in sample and not np.array_equal(y, sample["label"].astype(int).to_numpy()):
    raise ValueError("Label and status disagree")
groups = sample["project_name"].astype(str).to_numpy()

print("UNSEEN-PROJECT 5-FOLD SPLIT AUDIT — 5,710 UNIQUE BUILDS")
print("Each fold: 12 train projects / 4 validation projects / 4 test projects")
print("NOT chronological. No model training or test evaluation.")

splits = []
outer = GroupKFold(n_splits=5)
seen_test_projects = []
for fold_id, (trainval_idx, test_idx) in enumerate(outer.split(sample, y, groups), start=1):
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42 + fold_id)
    inner_train, inner_val = next(
        inner.split(sample.iloc[trainval_idx], y[trainval_idx], groups[trainval_idx])
    )
    train_idx, val_idx = trainval_idx[inner_train], trainval_idx[inner_val]
    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    project_sets = {name: set(groups[idx]) for name, idx in parts.items()}
    if any(project_sets[a] & project_sets[b] for a, b in
           [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError(f"Project overlap in fold {fold_id}")
    if [len(project_sets[k]) for k in ("train", "validation", "test")] != [12, 4, 4]:
        raise ValueError(f"Unexpected project counts in fold {fold_id}")
    seen_test_projects.extend(project_sets["test"])
    assignments = np.full(len(sample), "UNASSIGNED", dtype=object)
    for name, idx in parts.items():
        assignments[idx] = name
    if (assignments == "UNASSIGNED").any():
        raise ValueError(f"Unassigned rows in fold {fold_id}")
    frame = pd.DataFrame({
        "sample_row_id": np.arange(len(sample), dtype=int),
        "project_name": sample["project_name"].astype(str).to_numpy(),
        "build_id": sample["build_id"].astype(str).to_numpy(),
        "label": y,
        "fold": fold_id,
        "split": assignments,
    })
    splits.append(frame)
    print(f"\nFold {fold_id}:")
    for name in ("train", "validation", "test"):
        sub = frame.loc[frame["split"].eq(name)]
        print(f"  {name:10s}: builds={len(sub):4d}, projects={sub['project_name'].nunique():2d}, failure_ratio={sub['label'].mean():.4f}")
    print("  Held-out test projects:", ", ".join(sorted(project_sets["test"])))

if len(seen_test_projects) != 20 or len(set(seen_test_projects)) != 20:
    raise ValueError("Each of the 20 projects must appear as TEST in exactly one fold")
manifest = pd.concat(splits, ignore_index=True)
if len(manifest) != 5 * len(sample) or manifest.duplicated(["fold", "sample_row_id"]).any():
    raise ValueError("Manifest integrity check failed")
manifest.to_csv(OUTPUT_PATH, index=False)
print("\nAUDIT PASS: No project overlaps inside a fold; all projects tested exactly once.")
print("Manifest rows:", len(manifest))
print("Saved:", OUTPUT_PATH)
print("NEXT: Fix candidate feature policy and validation-only model selection before final test results.")

