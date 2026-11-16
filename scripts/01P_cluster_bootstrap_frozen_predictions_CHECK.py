"""Project-clustered bootstrap analysis of the FROZEN chronological test predictions.

Reads only pre-existing held-out predictions. No model loading/training, prediction
threshold selection, hyperparameter selection, or test-driven model changes.

Run in CICD_JSS_Strengthened:
    python 01P_cluster_bootstrap_frozen_predictions_CHECK.py

Interpretation: These descriptive percentile intervals resample the 20 sampled
repositories, taking all test builds from each selected repository. They quantify
project-level variability CONDITIONAL ON the chosen repositories and protocol;
they are not proof of generalization to all CI projects or causal effects.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
TRANSFORMERS = BASE / 'results_frozen_chronological_TEST_ONCE' / 'frozen_three_transformer_test_predictions.csv'
LOGREG = BASE / 'results_frozen_chronological_LOGREG_TEST_ONCE' / 'logreg_chronological_test_predictions.csv'
OUTPUT = BASE / 'results_frozen_chronological_PROJECT_BOOTSTRAP_CHECK'
SEED = 42
N_BOOTSTRAP = 3000
KEY = ['sample_row_id', 'project_name', 'build_id', 'true_label']

for path in (TRANSFORMERS, LOGREG):
    if not path.is_file():
        raise FileNotFoundError(f'Missing prediction file: {path}')

read_options = {'dtype': {'project_name': 'string', 'build_id': 'string'}, 'low_memory': False}
tf = pd.read_csv(TRANSFORMERS, **read_options)
lr = pd.read_csv(LOGREG, **read_options)

required_tf = KEY + [
    'distilbert_probability', 'distilbert_predicted_failure',
    'fixed_lora_probability', 'fixed_lora_predicted_failure',
    'optuna_lora_probability', 'optuna_lora_predicted_failure'
]
required_lr = KEY + ['failure_probability', 'predicted_failure']
for name, frame, required in [('transformer', tf, required_tf), ('logreg', lr, required_lr)]:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f'{name} file missing columns: {missing}')
    if frame[KEY[:3]].isna().any().any():
        raise ValueError(f'{name} file has missing build identity fields')
    if frame.duplicated(KEY[:3]).any():
        raise ValueError(f'{name} file repeats sample/project/build identity')

# Join on build identity (not row position), then independently verify true labels.
tf = tf.sort_values(KEY[:3]).reset_index(drop=True)
lr = lr.sort_values(KEY[:3]).reset_index(drop=True)
joined = tf.merge(lr, on=KEY[:3], how='outer', validate='one_to_one', indicator=True,
                  suffixes=('_tf', '_lr'))
if not joined['_merge'].eq('both').all():
    raise ValueError('The two prediction files contain different held-out builds.')
if not joined['true_label_tf'].eq(joined['true_label_lr']).all():
    raise ValueError('True labels disagree between the prediction files.')
if len(joined) != 1144 or joined['project_name'].nunique() != 20:
    raise ValueError(f'Unexpected test size/project count: {len(joined)} builds, '
                     f'{joined["project_name"].nunique()} projects')

labels = joined['true_label_tf'].to_numpy(dtype=int)
if not np.isin(labels, [0, 1]).all() or labels.sum() != 317:
    raise ValueError('Unexpected test labels: expected 317 failures among 1,144 builds')

specifications = [
    ('Logistic Regression', 'failure_probability', 'predicted_failure'),
    ('Full DistilBERT', 'distilbert_probability', 'distilbert_predicted_failure'),
    ('Fixed LoRA', 'fixed_lora_probability', 'fixed_lora_predicted_failure'),
    ('Optuna-LoRA', 'optuna_lora_probability', 'optuna_lora_predicted_failure'),
]
probabilities = {}
predictions = {}
for model, score_col, pred_col in specifications:
    p = pd.to_numeric(joined[score_col], errors='coerce').to_numpy(dtype=float)
    h = pd.to_numeric(joined[pred_col], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f'{model} has invalid predicted probabilities')
    if not np.isin(h, [0.0, 1.0]).all():
        raise ValueError(f'{model} has invalid binary predictions')
    probabilities[model] = p
    predictions[model] = h.astype(int)

projects = joined['project_name'].astype(str).to_numpy()
unique_projects = sorted(set(projects))
by_project = [np.flatnonzero(projects == project) for project in unique_projects]
models = [m for m, _, _ in specifications]
metric_names = ['F1', 'AP', 'ROC-AUC']


def score(y, pred, prob):
    if len(np.unique(y)) != 2:
        return (f1_score(y, pred, zero_division=0), np.nan, np.nan)
    return (f1_score(y, pred, zero_division=0),
            average_precision_score(y, prob), roc_auc_score(y, prob))


point = np.empty((len(models), 3), dtype=float)
for j, name in enumerate(models):
    point[j] = score(labels, predictions[name], probabilities[name])

expected = {  # From the previously frozen experiment logs, not fitted from this script.
    'Logistic Regression': (0.3742, 0.3212, 0.5371),
    'Full DistilBERT': (0.4018, 0.3425, 0.5412),
    'Fixed LoRA': (0.4216, 0.3013, 0.5372),
    'Optuna-LoRA': (0.3877, 0.3005, 0.5378),
}
for j, name in enumerate(models):
    if not np.allclose(point[j], expected[name], atol=0.00015, rtol=0):
        raise ValueError(f'{name}: scores do not reproduce frozen test results: '
                         f'{point[j]} versus {expected[name]}')

rng = np.random.default_rng(SEED)
boot = np.full((N_BOOTSTRAP, len(models), 3), np.nan, dtype=float)
base_f1 = np.empty(N_BOOTSTRAP)
prevalence = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    # Sample whole repositories, NOT independent rows. Repeated repositories
    # contribute their entire original test-build set each time sampled.
    sample_groups = rng.integers(0, len(by_project), size=len(by_project))
    rows = np.concatenate([by_project[g] for g in sample_groups])
    y = labels[rows]
    prevalence[b] = y.mean()
    base_f1[b] = f1_score(y, np.ones_like(y), zero_division=0)
    for j, model in enumerate(models):
        boot[b, j] = score(y, predictions[model][rows], probabilities[model][rows])


def interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 0.95 * N_BOOTSTRAP:
        raise ValueError('Too many bootstrap replicates with undefined metrics')
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)

results = []
for j, model in enumerate(models):
    r = {'model': model}
    for k, metric in enumerate(metric_names):
        lo, hi = interval(boot[:, j, k])
        r[f'{metric}_point'] = round(float(point[j, k]), 4)
        r[f'{metric}_CI_lower'] = lo
        r[f'{metric}_CI_upper'] = hi
    results.append(r)
result_frame = pd.DataFrame(results)

reference_f1 = f1_score(labels, np.ones_like(labels), zero_division=0)
contrasts = []

def add_contrast(description, metric, point_difference, bootstrap_difference):
    lo, hi = interval(bootstrap_difference)
    contrasts.append({'comparison': description, 'metric': metric,
                      'observed_difference': round(float(point_difference), 4),
                      'CI_lower': lo, 'CI_upper': hi})

for j, model in enumerate(models):
    add_contrast(f'{model} minus always-failure', 'F1',
                 point[j, 0] - reference_f1, boot[:, j, 0] - base_f1)
    add_contrast(f'{model} AP minus test failure prevalence', 'AP - prevalence',
                 point[j, 1] - labels.mean(), boot[:, j, 1] - prevalence)

for first, second, metric_index in [
    ('Fixed LoRA', 'Full DistilBERT', 0),
    ('Full DistilBERT', 'Logistic Regression', 1),
    ('Optuna-LoRA', 'Fixed LoRA', 0),
    ('Optuna-LoRA', 'Fixed LoRA', 1),
]:
    i, j = models.index(first), models.index(second)
    add_contrast(f'{first} minus {second}', metric_names[metric_index],
                 point[i, metric_index] - point[j, metric_index],
                 boot[:, i, metric_index] - boot[:, j, metric_index])
contrast_frame = pd.DataFrame(contrasts)

OUTPUT.mkdir(exist_ok=True)
result_frame.to_csv(OUTPUT / 'chronological_project_cluster_bootstrap_CI.csv', index=False)
contrast_frame.to_csv(OUTPUT / 'chronological_project_cluster_paired_differences_CI.csv', index=False)

print('=' * 78)
print('PROJECT-CLUSTERED BOOTSTRAP — FROZEN CHRONOLOGICAL TEST')
print('=' * 78)
print(f'Builds matched across files: {len(joined)}')
print(f'Projects resampled as complete clusters: {len(unique_projects)}')
print(f'Observed failed builds: {labels.sum()}')
print(f'Bootstrap replicates: {N_BOOTSTRAP}, random seed: {SEED}')
print('All frozen scores reproduced; no retraining or retuning.')
print('\n95% PERCENTILE INTERVALS (project-clustered; descriptive)')
for j, model in enumerate(models):
    parts = []
    for k, metric in enumerate(metric_names):
        lo, hi = interval(boot[:, j, k])
        parts.append(f'{metric}={point[j,k]:.4f} [{lo:.4f}, {hi:.4f}]')
    print(f'  {model}: ' + ' | '.join(parts))
print(f'\nAlways-failure F1 reference: {reference_f1:.4f}')
print(f'Test failure prevalence / AP reference: {labels.mean():.4f}')
print('\nPAIRED DIFFERENCES (same held-out builds)')
for c in contrasts:
    print(f'  {c["comparison"]} ({c["metric"]}): {c["observed_difference"]:+.4f} '
          f'[{c["CI_lower"]:+.4f}, {c["CI_upper"]:+.4f}]')
print(f'\nSaved new analysis tables to: {OUTPUT}')
print('No test predictions, model checkpoints, or original result folders changed.')
print('CAUTION: Bootstrap intervals reflect variability from resampling these 20 projects.')
print('Do not use these analyses to retune thresholds, select a different model, or claim causality.')

