"""Prepare a non-leaking *candidate metadata-text representation* from unique CI builds.

AUDIT ONLY. Does not train or score models; does not touch the test set for evaluation.
The metadata's real historical availability before build start is not established.
These strings are NOT CI logs and must not be used for build-progress prefixes.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'travis_unique_builds_20projects_CANDIDATE.csv'
SPLIT_PATH = ROOT / 'unique_builds_chronological_split_manifest.csv'
OUT_PATH = ROOT / 'unique_builds_three_feature_METADATA_TEXT_CANDIDATE.csv'

sample = pd.read_csv(DATA_PATH, dtype={'build_id': 'string'}, low_memory=False).reset_index(drop=True)
manifest = pd.read_csv(SPLIT_PATH, dtype={'build_id': 'string'}, low_memory=False)

required_data = {'project_name', 'build_id', 'status', 'branch', 'is_pr', 'num_commits'}
required_split = {'sample_row_id', 'project_name', 'build_id', 'label', 'split'}
if missing := required_data - set(sample.columns):
    raise ValueError(f'Missing sample columns: {sorted(missing)}')
if missing := required_split - set(manifest.columns):
    raise ValueError(f'Missing manifest columns: {sorted(missing)}')
if len(sample) != 5710 or len(manifest) != 5710:
    raise ValueError(f'Expected 5,710 rows: sample={len(sample)}, manifest={len(manifest)}')
if sample[['project_name', 'build_id']].isna().any().any():
    raise ValueError('Missing project/build ID')
if sample.duplicated(['project_name', 'build_id']).any():
    raise ValueError('Duplicate project/build ID in candidate sample')
if sample['project_name'].nunique() != 20:
    raise ValueError('Expected exactly 20 projects')
if manifest['sample_row_id'].isna().any() or manifest['sample_row_id'].duplicated().any():
    raise ValueError('Invalid sample_row_id in manifest')
manifest = manifest.set_index('sample_row_id').reindex(range(len(sample)))
if manifest[['project_name', 'build_id', 'label', 'split']].isna().any().any():
    raise ValueError('Missing or misaligned manifest rows')
for col in ['project_name', 'build_id']:
    if not (sample[col].astype(str).to_numpy() == manifest[col].astype(str).to_numpy()).all():
        raise ValueError(f'Sample and split do not align on {col}')

status = sample['status'].astype(str).str.strip().str.lower()
valid = {'passed', 'success', 'fixed', 'failed', 'errored'}
if not status.isin(valid).all():
    raise ValueError('Unknown/nonterminal status found')
labels = status.isin({'failed', 'errored'}).astype(int)
if 'label' in sample and not np.array_equal(labels.to_numpy(), sample['label'].astype(int).to_numpy()):
    raise ValueError('Sample label and status mismatch')
if not np.array_equal(labels.to_numpy(), manifest['label'].astype(int).to_numpy()):
    raise ValueError('Manifest label mismatch')
if not manifest['split'].isin(['train', 'validation', 'test']).all():
    raise ValueError('Unknown partition in split manifest')

# Explicit feature whitelist: no status, duration, test results, or project/build IDs
# enter the model's input text. Project/build IDs are kept ONLY as audit metadata.
def normalize_category(series):
    return series.astype('string').fillna('unknown').str.strip().replace('', 'unknown')

branch = normalize_category(sample['branch'])
is_pr = normalize_category(sample['is_pr']).str.lower()
counts_numeric = pd.to_numeric(sample['num_commits'], errors='coerce')
num_commits = counts_numeric.map(lambda v: f'{v:g}' if pd.notna(v) else 'unknown')

# Do not call the text a 'log': it consists of static feature labels and values.
metadata_text = ('branch ' + branch + ' is_pr ' + is_pr + ' num_commits ' + num_commits)

output = pd.DataFrame({
    'sample_row_id': range(len(sample)),
    'project_name': sample['project_name'],
    'build_id': sample['build_id'],
    'split': manifest['split'].to_numpy(),
    'metadata_text': metadata_text,
    'label': labels,
})
if output['metadata_text'].isna().any() or output['metadata_text'].str.len().eq(0).any():
    raise ValueError('Empty metadata text')

# A branch named 'fix-failure' may be legitimate if created before the build;
# flag outcome-related names for provenance review rather than deleting them.
suspicious_branch = branch.str.contains(
    r'(?i)(?:^|[^a-z])(?:fail|failed|failure|pass|passed|success|error|errored)(?:$|[^a-z])',
    regex=True, na=False
)
conflicting_text = (
    output.groupby(['project_name', 'metadata_text'])['label'].nunique().gt(1).sum()
)

print('=' * 72)
print('UNIQUE-BUILD THREE-FEATURE METADATA-TEXT AUDIT')
print('=' * 72)
print('Input unique builds:', len(sample))
print('Projects:', sample['project_name'].nunique())
print('Training rows:', int((output['split'] == 'train').sum()))
print('Validation rows:', int((output['split'] == 'validation').sum()))
print('Test rows (NOT evaluated):', int((output['split'] == 'test').sum()))
print('Model-input features: branch, is_pr, num_commits')
print('Missing/unknown branch:', int(branch.eq('unknown').sum()))
print('Missing/unknown is_pr:', int(is_pr.eq('unknown').sum()))
print('Missing/unknown num_commits:', int(num_commits.eq('unknown').sum()))
print('Branch names with outcome-like terms (manual provenance check):', int(suspicious_branch.sum()))
print('Project/text combinations with conflicting labels:', int(conflicting_text))
print('Average tokens per text:', round(float(metadata_text.str.split().str.len().mean()), 2))
output.to_csv(OUT_PATH, index=False)
print('Saved:', OUT_PATH)
print('AUDIT ONLY. Static metadata text is NOT a chronological CI build log.')
print('Do not interpret token prefixes as 25/50/75/100% build progress.')
print('Historical pre-build feature availability remains unverified.')

