"""Validation-only comparison of three models across five unseen-project folds.

Uses the 5,710-unique-build dataset and the project-disjoint manifest made by 01M.
For EACH fold: train on its 12 training projects, evaluate ONLY its four
validation projects, and leave its four test projects untouched.

Uses branch, is_pr, num_commits. Historical prediction-time provenance remains
unverified: these are provisional candidates, not proven leakage-free inputs.

Fold-wise model selection should be based on validation AP, and any eventual
outer-fold test scores must use model/threshold choices fixed per fold first.
No test-set scoring, manuscript values, or file modification occurs here.
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

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit('XGBoost missing: run python -m pip install xgboost') from exc

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / 'travis_unique_builds_20projects_CANDIDATE.csv'
MANIFEST_FILE = ROOT / 'unique_builds_unseen_project_5fold_manifest.csv'
for path in (SAMPLE_FILE, MANIFEST_FILE):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')

sample = pd.read_csv(SAMPLE_FILE, dtype={'build_id': 'string'}, low_memory=False).reset_index(drop=True)
manifest = pd.read_csv(MANIFEST_FILE, dtype={'build_id': 'string'}, low_memory=False)
required = {'sample_row_id', 'project_name', 'build_id', 'label', 'fold', 'split'}
if not required.issubset(manifest.columns):
    raise ValueError(f'Missing manifest columns: {sorted(required - set(manifest.columns))}')
if len(sample) != 5710 or sample['project_name'].nunique() != 20 or len(manifest) != len(sample) * 5:
    raise ValueError('Expected 5,710 unique builds across 20 projects and 28,550 manifest rows')
if sample.duplicated(['project_name', 'build_id']).any():
    raise ValueError('Repeated (project_name, build_id) in dataset')
if manifest.duplicated(['fold', 'sample_row_id']).any():
    raise ValueError('Repeated (fold, sample_row_id) in manifest')

status = sample['status'].astype(str).str.strip().str.lower()
if not status.isin(['passed', 'success', 'fixed', 'failed', 'errored']).all():
    raise ValueError('Unexpected outcome status')
y = status.isin(['failed', 'errored']).astype(int).to_numpy()
if 'label' in sample and not np.array_equal(y, sample['label'].astype(int).to_numpy()):
    raise ValueError('Dataset label differs from status')

# Three provisional candidate features. No outcome or post-build field is used.
X = pd.DataFrame({
    'branch': sample['branch'].fillna('unknown').astype(str),
    'is_pr': sample['is_pr'].fillna('unknown').astype(str),
    'num_commits': pd.to_numeric(sample['num_commits'], errors='coerce'),
})
projects = sample['project_name'].astype(str).to_numpy()
ids = sample['build_id'].astype(str).to_numpy()
all_test_projects = []
results = []

print('=' * 78)
print('UNSEEN-PROJECT 5-FOLD BASELINES — VALIDATION ONLY')
print('=' * 78)
print('Features: branch, is_pr, num_commits (historical timing provisional)')
print('Metric for fold-wise selection: validation average precision (AP)')
print('No test predictions, no test metrics, no files written.')

for fold in range(1, 6):
    frame = manifest.loc[manifest['fold'].eq(fold)].copy()
    if len(frame) != len(sample) or set(frame['sample_row_id']) != set(range(len(sample))):
        raise ValueError(f'Fold {fold}: missing or duplicate dataset row IDs')
    frame = frame.set_index('sample_row_id').reindex(range(len(sample)))
    if frame[['project_name', 'build_id', 'label', 'split']].isna().any().any():
        raise ValueError(f'Fold {fold}: missing manifest values')
    if not np.array_equal(frame['project_name'].astype(str).to_numpy(), projects):
        raise ValueError(f'Fold {fold}: project alignment mismatch')
    if not np.array_equal(frame['build_id'].astype(str).to_numpy(), ids):
        raise ValueError(f'Fold {fold}: build ID alignment mismatch')
    if not np.array_equal(frame['label'].astype(int).to_numpy(), y):
        raise ValueError(f'Fold {fold}: status/label mismatch')

    split = frame['split'].astype(str)
    if not split.isin(['train', 'validation', 'test']).all():
        raise ValueError(f'Fold {fold}: unrecognized partition')
    masks = {part: split.eq(part).to_numpy() for part in ['train', 'validation', 'test']}
    sets = {part: set(projects[mask]) for part, mask in masks.items()}
    if [len(sets[p]) for p in ('train', 'validation', 'test')] != [12, 4, 4]:
        raise ValueError(f'Fold {fold}: expected 12/4/4 projects')
    if (sets['train'] & sets['validation'] or sets['train'] & sets['test']
            or sets['validation'] & sets['test']):
        raise ValueError(f'Fold {fold}: project overlap')
    all_test_projects.extend(sets['test'])

    tr, va = masks['train'], masks['validation']
    yt, yv = y[tr], y[va]
    if len(np.unique(yt)) != 2 or len(np.unique(yv)) != 2:
        raise ValueError(f'Fold {fold}: missing positive/negative class in train/validation')

    # Preprocessing is fitted on this fold's training projects ONLY.
    prep = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['branch', 'is_pr']),
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), ['num_commits']),
    ])
    xtr = prep.fit_transform(X.loc[tr])
    xva = prep.transform(X.loc[va])

    neg, pos = int((yt == 0).sum()), int((yt == 1).sum())
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=neg / pos, tree_method='hist',
            random_state=42, n_jobs=-1,
        ),
    }
    print(f'\nFold {fold}: train={tr.sum()} builds, validation={va.sum()} builds, '
          f'test={masks["test"].sum()} builds (NOT evaluated); '
          f'validation failure fraction={yv.mean():.4f}', flush=True)
    for name, model in models.items():
        print(f'  Training {name} ...', flush=True)
        model.fit(xtr, yt)
        probs = model.predict_proba(xva)[:, 1]
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        options = [(f1_score(yv, probs >= t, zero_division=0), float(t))
                   for t in thresholds]
        best_f1, threshold = max(options, key=lambda z: (z[0], -z[1]))
        pred = (probs >= threshold).astype(int)
        results.append({
            'fold': fold,
            'model': name,
            'val_threshold': threshold,
            'val_precision': precision_score(yv, pred, zero_division=0),
            'val_recall': recall_score(yv, pred, zero_division=0),
            'val_f1_selected': best_f1,
            'val_ap': average_precision_score(yv, probs),
            'val_roc_auc': roc_auc_score(yv, probs),
        })

if len(all_test_projects) != 20 or len(set(all_test_projects)) != 20:
    raise ValueError('Test-project coverage incorrect across folds')

r = pd.DataFrame(results)
print('\n' + '=' * 78)
print('FOLD-BY-FOLD VALIDATION METRICS (NOT PUBLICATION RESULTS)')
print('=' * 78)
print(r.to_string(index=False, formatters={
    k: '{:.4f}'.format for k in ['val_precision', 'val_recall',
                                 'val_f1_selected', 'val_ap', 'val_roc_auc']
}))
print('\n' + '=' * 78)
print('MEAN VALIDATION RESULTS ACROSS FIVE FOLDS — DESCRIPTIVE ONLY')
print('=' * 78)
summary = r.groupby('model', sort=False)[['val_f1_selected', 'val_ap', 'val_roc_auc']].agg(['mean', 'std'])
print(summary.round(4).to_string())
print('\nPER-FOLD VALIDATION-AP WINNERS (FOR LATER, FOLD-SPECIFIC MODEL CHOICE)')
for fold, df in r.groupby('fold'):
    winner = df.sort_values(['val_ap', 'val_f1_selected'], ascending=False).iloc[0]
    print(f'  Fold {fold}: {winner["model"]}, AP={winner["val_ap"]:.4f}, '
          f'validation-F1 threshold={winner["val_threshold"]:.2f}')
print('\nNo test predictions generated. No files modified.')
print('Do not use these validation-selected F1 values as final test results.')
print('Note: project-disjoint folds are NOT chronological.')
print('CAUTION: input availability before build execution remains unverified.')

