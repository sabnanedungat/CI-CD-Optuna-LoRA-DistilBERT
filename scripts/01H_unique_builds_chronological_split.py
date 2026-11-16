"""Create an AUDIT-ONLY chronological split for the new 5,710 unique-build sample.

Do not use the previous 6,000-row split manifests with this dataset.
Does not train models or modify the source CSV.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "travis_unique_builds_20projects_CANDIDATE.csv"
OUTPUT = ROOT / "unique_builds_chronological_split_manifest.csv"

if not SOURCE.exists():
    raise FileNotFoundError(f"Not found: {SOURCE}")

# Keep build IDs as strings to avoid accidental numeric coercion.
df = pd.read_csv(SOURCE, low_memory=False, dtype={"build_id": "string"})
required = {"project_name", "build_id", "started_at", "status"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")
if df[list(required)].isna().any().any():
    raise ValueError("Source contains missing project, build ID, timestamp or status")
if df.duplicated(["project_name", "build_id"]).any():
    raise ValueError("Duplicate project/build IDs found; stop and inspect")

status = df["status"].astype(str).str.strip().str.lower()
valid = {"passed", "success", "fixed", "failed", "errored"}
if not status.isin(valid).all():
    raise ValueError("Unexpected status values")
df["label_checked"] = status.isin({"failed", "errored"}).astype(int)
if "label" in df.columns and not df["label"].astype(int).equals(df["label_checked"]):
    raise ValueError("Original label does not agree with status")

df["time_checked"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
if df["time_checked"].isna().any():
    raise ValueError("Some start timestamps could not be parsed")

df["sample_row_id"] = range(len(df))
assignments = {}
for project, part in df.groupby("project_name", sort=True):
    # Tied timestamps stay together, preventing equal-time records from
    # being split across train, validation, and test.
    part = part.sort_values(["time_checked", "build_id"], kind="stable")
    timestamps = part["time_checked"].drop_duplicates().tolist()
    n_times = len(timestamps)
    if n_times < 5:
        raise ValueError(f"Not enough distinct timestamps for project: {project}")
    cut1 = max(1, min(n_times - 2, round(n_times * 0.64)))
    cut2 = max(cut1 + 1, min(n_times - 1, round(n_times * 0.80)))
    train_end = timestamps[cut1 - 1]
    val_end = timestamps[cut2 - 1]
    for row in part.itertuples(index=False):
        split = ("train" if row.time_checked <= train_end else
                 "validation" if row.time_checked <= val_end else "test")
        assignments[row.sample_row_id] = split

manifest = pd.DataFrame({
    "sample_row_id": df["sample_row_id"],
    "project_name": df["project_name"],
    "build_id": df["build_id"],
    "started_at": df["started_at"],
    "label": df["label_checked"],
    "split": df["sample_row_id"].map(assignments),
})

assert not manifest["split"].isna().any()
assert set(manifest["split"]) == {"train", "validation", "test"}
assert len(manifest) == len(df)
assert manifest.groupby(["project_name", "build_id"])["split"].nunique().max() == 1

for project, part in df.groupby("project_name"):
    order = manifest.loc[part.index, "split"]
    tm = part["time_checked"]
    for before, after in [("train", "validation"), ("validation", "test")]:
        early, late = tm[order == before], tm[order == after]
        if early.empty or late.empty or early.max() >= late.min():
            raise ValueError(f"Chronological separation failure for {project}")

print("UNIQUE-BUILD CHRONOLOGICAL SPLIT AUDIT")
print("=" * 65)
for split in ["train", "validation", "test"]:
    p = manifest[manifest["split"] == split]
    print(f"{split:10s}: rows={len(p):4d}, projects={p['project_name'].nunique():2d}, "
          f"failure_ratio={p['label'].mean():.4f}")
print("Unique builds:", len(manifest))
print("Build overlap across splits: 0")
print("Within-project chronological ordering: PASS")
manifest.to_csv(OUTPUT, index=False)
print("Saved:", OUTPUT)
print("AUDIT ONLY: old split manifests cannot be reused. Do not train yet.")
print("Feature availability at build-start remains to be verified.")

