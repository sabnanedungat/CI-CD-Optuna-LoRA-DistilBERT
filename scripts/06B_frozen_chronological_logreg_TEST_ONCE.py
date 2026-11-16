"""One-time chronological test of the PREVIOUSLY validation-selected Logistic Regression baseline.

Exact feature engineering/model configuration from 01L_no_language_three_baselines_VALIDATION.py.
Frozen classification threshold: 0.54 (selected before any chronological test evaluation).
Refits ONLY on the original chronological TRAIN partition because the original
validation script did not save a checkpoint. Does not tune hyperparameters or
use test labels to choose anything. This is NOT an early-log experiment.

Historical prediction-time availability of branch, is_pr, num_commits remains provisional.
"""

from pathlib import Path
import json
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
MANIFEST_FILE = ROOT / 'unique_builds_chronological_split_manifest.csv'
OUTPUT_DIR = ROOT / 'results_frozen_chronological_LOGREG_TEST_ONCE'
FROZEN_THRESHOLD = 0.54
EXPECTED_VALIDATION_AP = 0.4423
EXPECTED_VALIDATION_F1 = 0.4647


def main():
    for file in (SAMPLE_FILE, MANIFEST_FILE):
        if not file.is_file():
            raise FileNotFoundError(f'Missing required data file: {file}')
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f'Output directory already exists: {OUTPUT_DIR}. '
            'This one-time test script will not overwrite results.'
        )

    sample = pd.read_csv(SAMPLE_FILE, dtype={'build_id': 'string'}, low_memory=False).reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST_FILE, dtype={'build_id': 'string'}, low_memory=False)
    required = {'sample_row_id', 'project_name', 'build_id', 'label', 'split'}
    if required - set(manifest.columns):
        raise ValueError(f'Missing manifest columns: {required - set(manifest.columns)}')
    if len(sample) != 5710 or len(manifest) != 5710:
        raise ValueError('Expected exactly 5710 rows in both datasets')
    if manifest['sample_row_id'].duplicated().any():
        raise ValueError('Duplicate sample_row_id in manifest')
    manifest = manifest.set_index('sample_row_id').reindex(range(5710))
    if manifest[['project_name', 'build_id', 'label', 'split']].isna().any().any():
        raise ValueError('Missing or misaligned manifest data')
    for col in ('project_name', 'build_id'):
        if not sample[col].astype(str).equals(manifest[col].astype(str)):
            raise ValueError(f'Mismatch between sample and manifest: {col}')
    if sample.duplicated(['project_name', 'build_id']).any():
        raise ValueError('Duplicate CI builds')
    status = sample['status'].astype(str).str.strip().str.lower()
    if not status.isin({'passed', 'success', 'fixed', 'failed', 'errored'}).all():
        raise ValueError('Unexpected status in sample')
    y = status.isin(['failed', 'errored']).astype(int).reset_index(drop=True)
    if not np.array_equal(y.to_numpy(), manifest['label'].astype(int).to_numpy()):
        raise ValueError('Manifest and sample outcome labels do not match')
    split = manifest['split'].astype(str)
    if not split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Unexpected split value')
    train = split.eq('train').to_numpy()
    val = split.eq('validation').to_numpy()
    test = split.eq('test').to_numpy()
    if (int(train.sum()), int(val.sum()), int(test.sum())) != (3651, 915, 1144):
        raise ValueError('Chronological split row counts do not match the frozen protocol')

    dates = pd.to_datetime(sample['started_at'], errors='coerce', utc=True)
    if dates.isna().any():
        raise ValueError('Missing build-start timestamp')
    for project, idx in sample.groupby('project_name').groups.items():
        rows = list(idx)
        for earlier, later in (('train', 'validation'), ('validation', 'test')):
            old = dates.iloc[rows][split.iloc[rows].eq(earlier)]
            new = dates.iloc[rows][split.iloc[rows].eq(later)]
            if len(old) and len(new) and old.max() > new.min():
                raise ValueError(f'Chronological order violation in project {project}')

    # Identical input and preprocessing to 01L (only 3 provisional candidate features).
    X = pd.DataFrame({
        'branch': sample['branch'].fillna('unknown').astype(str),
        'is_pr': sample['is_pr'].fillna('unknown').astype(str),
        'num_commits': pd.to_numeric(sample['num_commits'], errors='coerce'),
    })
    preprocessing = ColumnTransformer([
        ('cats', OneHotEncoder(handle_unknown='ignore'), ['branch', 'is_pr']),
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), ['num_commits']),
    ])
    X_train = preprocessing.fit_transform(X.loc[train])
    X_validation = preprocessing.transform(X.loc[val])
    y_train = y.loc[train].to_numpy()
    y_validation = y.loc[val].to_numpy()

    print('=' * 75)
    print('FROZEN LOGISTIC REGRESSION — CHRONOLOGICAL TEST ONCE')
    print('=' * 75)
    print('Features: branch, is_pr, num_commits (prediction-time timing provisional)')
    print('Original model specification: 01L_no_language_three_baselines_VALIDATION.py')
    print('Train/validation/test rows: 3651 / 915 / 1144')
    print(f'Threshold frozen from earlier validation: {FROZEN_THRESHOLD:.2f}')
    print('Refitting on train only because 01L did not save a checkpoint; no retuning.')

    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    pval = model.predict_proba(X_validation)[:, 1]
    val_ap = float(average_precision_score(y_validation, pval))
    val_f1 = float(f1_score(y_validation, pval >= FROZEN_THRESHOLD, zero_division=0))
    print(f'Validation reproduction: AP={val_ap:.4f}, F1@0.54={val_f1:.4f}')
    if (abs(val_ap - EXPECTED_VALIDATION_AP) > 0.00015
            or abs(val_f1 - EXPECTED_VALIDATION_F1) > 0.00015):
        raise RuntimeError(
            'STOP: frozen validation results not reproduced. '
            'The test partition has NOT been evaluated.'
        )

    # Only after the validation reproduction passes do we transform/score test.
    X_test = preprocessing.transform(X.loc[test])
    yt = y.loc[test].to_numpy()
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= FROZEN_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel().tolist()
    results = {
        'model': 'Logistic Regression',
        'protocol': '5710_unique_builds_within_project_chronological',
        'features': ['branch', 'is_pr', 'num_commits'],
        'feature_timing_proven': False,
        'threshold_frozen_on_validation': FROZEN_THRESHOLD,
        'train_builds': int(train.sum()),
        'validation_builds': int(val.sum()),
        'test_builds': int(test.sum()),
        'validation_ap_reproduced': val_ap,
        'validation_f1_at_frozen_threshold_reproduced': val_f1,
        'test_failure_prevalence': float(yt.mean()),
        'test_accuracy': float(accuracy_score(yt, pred)),
        'test_precision': float(precision_score(yt, pred, zero_division=0)),
        'test_recall': float(recall_score(yt, pred, zero_division=0)),
        'test_f1': float(f1_score(yt, pred, zero_division=0)),
        'test_ap': float(average_precision_score(yt, proba)),
        'test_roc_auc': float(roc_auc_score(yt, proba)),
        'test_predicted_failure_fraction': float(pred.mean()),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'always_failure_test_f1': float(f1_score(yt, np.ones_like(yt))),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({
        'sample_row_id': manifest.index[test].to_numpy(),
        'project_name': sample.loc[test, 'project_name'].to_numpy(),
        'build_id': sample.loc[test, 'build_id'].astype(str).to_numpy(),
        'true_label': yt, 'failure_probability': proba,
        'predicted_failure': pred,
    }).to_csv(OUTPUT_DIR / 'logreg_chronological_test_predictions.csv', index=False)
    (OUTPUT_DIR / 'logreg_chronological_test_metrics.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8'
    )

    print('\n' + '=' * 75)
    print('FROZEN CHRONOLOGICAL LOGREG TEST — RESULTS')
    print('=' * 75)
    for key in ('test_failure_prevalence', 'test_accuracy', 'test_precision',
                'test_recall', 'test_f1', 'test_ap', 'test_roc_auc',
                'test_predicted_failure_fraction', 'tn', 'fp', 'fn', 'tp',
                'always_failure_test_f1'):
        value = results[key]
        print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
    print('New results folder:', OUTPUT_DIR)
    print('No model choices or thresholds updated based on test outcomes.')
    print('Reminder: these are static metadata features, NOT progressive CI execution logs.')


if __name__ == '__main__':
    main()

