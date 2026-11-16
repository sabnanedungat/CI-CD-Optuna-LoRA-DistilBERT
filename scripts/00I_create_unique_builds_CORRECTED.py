"""
Correction 1 / Step 20: Select up to 300 UNIQUE CI builds per existing project.
AUDIT/CANDIDATE ONLY -- DOES NOT OVERWRITE original datasets or split manifests.
If same-build records disagree about label/features/time, STOP for inspection.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'travistorrent.csv'
PROJECTS = ROOT / 'selected_projects_summary.csv'
OUT = ROOT / 'travis_unique_builds_20projects_CANDIDATE.csv'
TARGET_PER_PROJECT = 300

for file in (RAW, PROJECTS):
    if not file.exists():
        raise FileNotFoundError(f'Missing required file: {file}')

names = pd.read_csv(PROJECTS)['project_name'].dropna().astype(str).tolist()
if len(names) != 20 or len(set(names)) != 20:
    raise ValueError('Expected exactly 20 distinct projects in selected_projects_summary.csv')

cols = [
    'project_name', 'build_id', 'started_at', 'status',
    'lang', 'branch', 'is_pr', 'num_commits', 'num_committers'
]
parts = []
for part in pd.read_csv(RAW, usecols=cols, chunksize=50000,
                        dtype={'build_id': 'string'}, low_memory=False):
    selection = part.loc[part['project_name'].isin(names)]
    if not selection.empty:
        parts.append(selection.copy())
if not parts:
    raise ValueError('No matching project records')

data = pd.concat(parts, ignore_index=True)
print('Records for selected projects:', len(data))
print('Rows missing build ID:', int(data['build_id'].isna().sum()))
print('Rows missing timestamps:', int(data['started_at'].isna().sum()))
data = data.dropna(subset=['build_id', 'status', 'started_at']).copy()
data['status'] = data['status'].astype(str).str.strip().str.lower()
valid = {'passed', 'success', 'fixed', 'failed', 'errored'}
unknown_status = (~data['status'].isin(valid)).sum()
print('Rows with unknown/nonterminal status excluded:', int(unknown_status))
data = data.loc[data['status'].isin(valid)].copy()
data['label'] = data['status'].isin(['failed', 'errored']).astype(int)
keys = ['project_name', 'build_id']

# Check consistency among all records of each CI build.
must_agree = ['label', 'started_at', 'lang', 'branch',
              'is_pr', 'num_commits', 'num_committers']
grouped = data.groupby(keys, dropna=False)
disagree = grouped[must_agree].nunique(dropna=False).gt(1)
print('\nWithin-build consistency checks:')
for field in must_agree:
    print(f'  {field}: {int(disagree[field].sum())} inconsistent build groups')
if disagree.any(axis=1).any():
    print('\nSTOP: Some records in a build disagree on a checked field.')
    print('We need to inspect their meaning before choosing one build-level record.')
    raise SystemExit(2)

unique = data.drop_duplicates(keys, keep='first').copy()
counts = unique.groupby('project_name').size()
print('\nUnique builds available in each selected project:')
print(counts.to_string())
shortfall = counts[counts < TARGET_PER_PROJECT]
if not shortfall.empty:
    print('\nProjects with fewer than 300 eligible unique builds (keeping all):')
    print(shortfall.to_string())

sample_parts = []
for project in names:
    group = unique.loc[unique['project_name'] == project]
    sample_size = min(TARGET_PER_PROJECT, len(group))
    if sample_size == 0:
        raise ValueError(f'No eligible unique builds for {project}')
    sample_parts.append(group.sample(n=sample_size, random_state=42))
result = pd.concat(sample_parts, ignore_index=True)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)
expected_rows = sum(min(TARGET_PER_PROJECT, int(counts[project])) for project in names)
assert len(result) == expected_rows
assert result['project_name'].nunique() == 20
assert not result.duplicated(keys).any()
result.to_csv(OUT, index=False)

print('\nUNIQUE-BUILD SAMPLE CREATED')
print('Rows:', len(result))
print('Unique project/build_id pairs:', result[keys].drop_duplicates().shape[0])
print('Projects:', result['project_name'].nunique())
print('Per-project sample sizes:')
print(result['project_name'].value_counts().sort_index().to_string())
print('Failed/errored:', int(result['label'].sum()))
print('Passed/success/fixed:', int((result['label'] == 0).sum()))
print('Saved:', OUT)
print('CANDIDATE ONLY: do not train yet; pre-build feature timing remains unverified.')
print('Existing split manifests do NOT align with this new sample.')

