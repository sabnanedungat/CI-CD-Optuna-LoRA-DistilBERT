"""One-time, frozen Logistic Regression unseen-project evaluation.

Requires the same 5,710 unique-build sample and 5-fold project-disjoint
manifest as scripts 01M/01N. Training-only preprocessing per fold.

Before examining test results, model/feature selection was made from 01N:
  features: branch, is_pr, num_commits
  model: LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
  metric for model choice: fold-wise validation average precision (AP)
  F1 threshold per fold (selected from validation): 0.34,0.23,0.26,0.23,0.05

This script does not tune from test, write files, or change existing data.
Predictions are provisional: historical feature availability before build
execution has not been conclusively verified.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / 'travis_unique_builds_20projects_CANDIDATE.csv'
MANIFEST_FILE = ROOT / 'unique_builds_unseen_project_5fold_manifest.csv'
THRESHOLDS_FROM_01N = {1: .34, 2: .23, 3: .26, 4: .23, 5: .05}

for path in (SAMPLE_FILE, MANIFEST_FILE):
    if not path.is_file():
        raise FileNotFoundError('Missing required file: ' + str(path))
sample = pd.read_csv(SAMPLE_FILE, dtype={'build_id': 'string'}, low_memory=False).reset_index(drop=True)
manifest = pd.read_csv(MANIFEST_FILE, dtype={'build_id': 'string'}, low_memory=False)
required = {'sample_row_id', 'project_name', 'build_id', 'label', 'fold', 'split'}
if not required.issubset(manifest.columns):
    raise ValueError('Manifest missing columns: ' + str(required - set(manifest.columns)))
if len(sample) != 5710 or len(manifest) != 28550 or sample['project_name'].nunique() != 20:
    raise ValueError('Unexpected dataset or manifest size')
if sample.duplicated(['project_name', 'build_id']).any():
    raise ValueError('Duplicate build IDs in unique-build dataset')
if manifest.duplicated(['fold', 'sample_row_id']).any():
    raise ValueError('Duplicate fold/sample_row_id values')
if set(manifest['fold'].unique()) != set(THRESHOLDS_FROM_01N):
    raise ValueError('Unexpected folds in manifest')

status = sample['status'].astype(str).str.strip().str.lower()
if not status.isin(['passed', 'success', 'fixed', 'failed', 'errored']).all():
    raise ValueError('Unexpected status values')
y = status.isin(['failed', 'errored']).astype(int).to_numpy()
if 'label' in sample and not np.array_equal(y, sample['label'].astype(int).to_numpy()):
    raise ValueError('Sample label/status mismatch')
projects = sample['project_name'].astype(str).to_numpy()
ids = sample['build_id'].astype(str).to_numpy()
X = pd.DataFrame({
    'branch': sample['branch'].fillna('unknown').astype(str),
    'is_pr': sample['is_pr'].fillna('unknown').astype(str),
    'num_commits': pd.to_numeric(sample['num_commits'], errors='coerce'),
})

print('=' * 76)
print('FROZEN LOGISTIC REGRESSION — FIVE UNSEEN-PROJECT TEST FOLDS')
print('=' * 76)
print('Features: branch, is_pr, num_commits (historical availability provisional)')
print('Thresholds frozen from validation:', THRESHOLDS_FROM_01N)
print('NO hyperparameter search, model reselection, or test-driven tuning.')
print('The five test folds are not chronological.')

results = []
all_test_projects = []
all_test_row_ids = []
for fold in range(1, 6):
    f = manifest.loc[manifest['fold'].eq(fold)].copy()
    if len(f) != len(sample) or set(f['sample_row_id']) != set(range(len(sample))):
        raise ValueError(f'Fold {fold}: incorrect row coverage')
    f = f.set_index('sample_row_id').reindex(range(len(sample)))
    if f[['project_name', 'build_id', 'label', 'split']].isna().any().any():
        raise ValueError(f'Fold {fold}: missing manifest values')
    if not np.array_equal(f['project_name'].astype(str).to_numpy(), projects):
        raise ValueError(f'Fold {fold}: project alignment mismatch')
    if not np.array_equal(f['build_id'].astype(str).to_numpy(), ids):
        raise ValueError(f'Fold {fold}: build ID mismatch')
    if not np.array_equal(f['label'].astype(int).to_numpy(), y):
        raise ValueError(f'Fold {fold}: label mismatch')
    splits = f['split'].astype(str)
    if not splits.isin(['train', 'validation', 'test']).all():
        raise ValueError(f'Fold {fold}: unexpected split')
    tr = splits.eq('train').to_numpy()
    va = splits.eq('validation').to_numpy()
    te = splits.eq('test').to_numpy()
    sets = {name: set(projects[mask]) for name, mask in
            [('train', tr), ('validation', va), ('test', te)]}
    if [len(sets[n]) for n in ['train', 'validation', 'test']] != [12, 4, 4]:
        raise ValueError(f'Fold {fold}: wrong train/validation/test project counts')
    if any(sets[a] & sets[b] for a, b in
           [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError(f'Fold {fold}: project overlap')
    all_test_projects.extend(sets['test'])
    all_test_row_ids.extend(np.flatnonzero(te).tolist())
    if np.unique(y[tr]).size != 2 or np.unique(y[te]).size != 2:
        raise ValueError(f'Fold {fold}: train/test needs both classes')

    preprocess = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['branch', 'is_pr']),
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), ['num_commits']),
    ])
    model = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42)),
    ])
    model.fit(X.loc[tr], y[tr])
    probabilities = model.predict_proba(X.loc[te])[:, 1]
    ytest = y[te]
    threshold = THRESHOLDS_FROM_01N[fold]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ytest, predictions, labels=[0, 1]).ravel()
    results.append({
        'fold': fold,
        'test_builds': int(te.sum()),
        'test_failure_ratio': ytest.mean(),
        'fixed_threshold': threshold,
        'test_precision': precision_score(ytest, predictions, zero_division=0),
        'test_recall': recall_score(ytest, predictions, zero_division=0),
        'test_f1': f1_score(ytest, predictions, zero_division=0),
        'test_ap': average_precision_score(ytest, probabilities),
        'test_roc_auc': roc_auc_score(ytest, probabilities),
        'ap_minus_prevalence': average_precision_score(ytest, probabilities) - ytest.mean(),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    })
    print(f'Fold {fold} evaluated on {te.sum()} previously held-out test builds.', flush=True)

if len(all_test_projects) != 20 or len(set(all_test_projects)) != 20:
    raise ValueError('Test project coverage is not exactly once')
if len(all_test_row_ids) != len(sample) or len(set(all_test_row_ids)) != len(sample):
    raise ValueError('Each build should be evaluated as test exactly once')

r = pd.DataFrame(results)
print('\n' + '=' * 76)
print('FINAL HELD-OUT TEST SCORES UNDER THE FROZEN CANDIDATE PROTOCOL')
print('=' * 76)
formatters = {k: '{:.4f}'.format for k in [
    'test_failure_ratio', 'test_precision', 'test_recall',
    'test_f1', 'test_ap', 'test_roc_auc', 'ap_minus_prevalence'
]}
formatters['fixed_threshold'] = '{:.2f}'.format
print(r.to_string(index=False, formatters=formatters))
print('\nFOLD-LEVEL MEAN AND SAMPLE STANDARD DEVIATION (n=5):')
for key in ['test_precision', 'test_recall', 'test_f1', 'test_ap',
            'test_roc_auc', 'ap_minus_prevalence']:
    print(f'  {key}: {r[key].mean():.4f} +/- {r[key].std(ddof=1):.4f}')
print('\nTest projects covered exactly once; no files written.')
print('CAUTION: Feature provenance not conclusively established; do not claim a')
print('leakage-free early-warning model or finalize manuscript values yet.')
print('CAUTION: Do not choose features/models/thresholds using these test scores.')

