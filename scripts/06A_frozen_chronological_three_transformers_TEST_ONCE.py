"""One-time, FROZEN chronological test evaluation of THREE saved transformer models.

Place this file next to:
  unique_builds_three_feature_METADATA_TEXT_CANDIDATE.csv
  unique_builds_chronological_split_manifest.csv
  results_distilbert_unique_chrono_VALIDATION_ONLY/
  results_lora_unique_chrono_VALIDATION_ONLY/
  results_optuna_lora_unique_chrono_VALIDATION_ONLY/

This program does NOT train, tune, evaluate validation data, or change prior results.
It saves test metrics/predictions in a NEW directory and refuses to overwrite it.

Scientific limitation: input text is static build metadata, NOT CI execution logs;
historical availability of branch/is_pr/num_commits at build start is provisional.
"""

from pathlib import Path
import gc
import json
import math

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

ROOT = Path(__file__).resolve().parent
METADATA_CSV = ROOT / 'unique_builds_three_feature_METADATA_TEXT_CANDIDATE.csv'
MANIFEST_CSV = ROOT / 'unique_builds_chronological_split_manifest.csv'
OUTPUT_DIR = ROOT / 'results_frozen_chronological_TEST_ONCE'
MODEL_ID = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
SEED = 42
EXPECTED_FEATURES = ['branch', 'is_pr', 'num_commits']
EXPECTED_DATASET = '5710_unique_builds_within_project_chronological'
EXPECTED_INPUT_TYPE = 'static_metadata_text_not_execution_log'

# All values frozen BEFORE reading the held-out chronological test labels.
# These were selected using validation data in the earlier 03B/04C/05B runs.
MODELS = [
    {
        'name': 'Full DistilBERT',
        'kind': 'full',
        'dir': ROOT / 'results_distilbert_unique_chrono_VALIDATION_ONLY',
        'checkpoint': 'best_model',
        'summary_threshold_key': 'validation_threshold_selected_by_F1',
        'frozen_threshold': 0.49,
        'frozen_epoch': 5,
    },
    {
        'name': 'Fixed LoRA-DistilBERT',
        'kind': 'adapter',
        'dir': ROOT / 'results_lora_unique_chrono_VALIDATION_ONLY',
        'checkpoint': 'best_adapter',
        'summary_threshold_key': 'validation_threshold_selected_by_F1',
        'frozen_threshold': 0.35,
        'frozen_epoch': 4,
    },
    {
        'name': 'Optuna-LoRA-DistilBERT',
        'kind': 'adapter',
        'dir': ROOT / 'results_optuna_lora_unique_chrono_VALIDATION_ONLY',
        'checkpoint': 'best_adapter',
        'summary_threshold_key': 'validation_selected_threshold',
        'frozen_threshold': 0.47,
        'frozen_epoch': 4,
    },
]


class TestTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokens = tokenizer(
            texts, truncation=True, padding=False, max_length=MAX_LENGTH
        )

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        return {key: values[idx] for key, values in self.tokens.items()}


def check_dataset():
    for file in (METADATA_CSV, MANIFEST_CSV):
        if not file.is_file():
            raise FileNotFoundError(f'Required CSV not found: {file}')
    data = pd.read_csv(METADATA_CSV, dtype={'build_id': 'string'}, low_memory=False)
    manifest = pd.read_csv(MANIFEST_CSV, dtype={'build_id': 'string'}, low_memory=False)
    common = {'sample_row_id', 'project_name', 'build_id', 'split', 'label'}
    if missing := (common | {'metadata_text'}) - set(data.columns):
        raise ValueError(f'Metadata CSV missing columns: {sorted(missing)}')
    if missing := common - set(manifest.columns):
        raise ValueError(f'Split manifest missing columns: {sorted(missing)}')
    if (len(data), len(manifest)) != (5710, 5710):
        raise ValueError('Expected exactly 5710 rows in both CSV files')
    if data['sample_row_id'].duplicated().any() or manifest['sample_row_id'].duplicated().any():
        raise ValueError('Duplicate sample_row_id')
    if data.duplicated(['project_name', 'build_id']).any():
        raise ValueError('Duplicate project/build_ID pair')
    data = data.set_index('sample_row_id').reindex(range(5710))
    manifest = manifest.set_index('sample_row_id').reindex(range(5710))
    for field in ('project_name', 'build_id', 'split', 'label'):
        if data[field].isna().any() or manifest[field].isna().any():
            raise ValueError(f'Missing or misaligned {field}')
        if not np.array_equal(
            data[field].astype(str).to_numpy(), manifest[field].astype(str).to_numpy()
        ):
            raise ValueError(f'Metadata and manifest disagree on {field}')
    if data['metadata_text'].isna().any() or data['metadata_text'].astype(str).str.strip().eq('').any():
        raise ValueError('Empty/missing metadata_text')
    if not data['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unexpected split assignment')
    if not data['label'].isin([0, 1]).all():
        raise ValueError('Unexpected label')
    if data['project_name'].nunique() != 20:
        raise ValueError('Expected 20 projects')
    counts = data['split'].value_counts()
    if any(int(counts.get(s, 0)) != n for s, n in (
        ('train', 3651), ('validation', 915), ('test', 1144)
    )):
        raise ValueError('Chronological split sizes disagree with the audited manifest')
    # Preserve original CSV row order for prediction export and group coverage.
    test = data.loc[data['split'].eq('test')].copy()
    if test['label'].nunique() != 2:
        raise ValueError('Test split unexpectedly contains only one class')
    return test


def check_model_configs():
    for spec in MODELS:
        summary_path = spec['dir'] / 'validation_summary.json'
        checkpoint = spec['dir'] / spec['checkpoint']
        if not summary_path.is_file():
            raise FileNotFoundError(f'Missing frozen validation summary: {summary_path}')
        if not checkpoint.is_dir():
            raise FileNotFoundError(f'Missing saved model checkpoint: {checkpoint}')
        needed_config = 'config.json' if spec['kind'] == 'full' else 'adapter_config.json'
        if not (checkpoint / needed_config).is_file():
            raise FileNotFoundError(f'Missing {needed_config} in {checkpoint}')
        info = json.loads(summary_path.read_text(encoding='utf-8'))
        if info.get('dataset') != EXPECTED_DATASET:
            raise ValueError(f'{spec["name"]}: unexpected dataset in saved summary')
        if info.get('features') != EXPECTED_FEATURES:
            raise ValueError(f'{spec["name"]}: unexpected features in saved summary')
        if info.get('input_type') != EXPECTED_INPUT_TYPE:
            raise ValueError(f'{spec["name"]}: unexpected input representation')
        for key, expected in (('train_builds', 3651), ('validation_builds', 915),
                              ('test_builds_not_evaluated', 1144)):
            if info.get(key) != expected:
                raise ValueError(f'{spec["name"]}: {key} differs from previous run')
        saved_threshold = info.get(spec['summary_threshold_key'])
        if saved_threshold is None or not math.isclose(
            float(saved_threshold), spec['frozen_threshold'], abs_tol=1e-8
        ):
            raise ValueError(
                f'{spec["name"]}: threshold mismatch. Expected '
                f'{spec["frozen_threshold"]}, found {saved_threshold}. STOP; do not retune.'
            )
        epoch_key = 'best_final_epoch' if spec['name'].startswith('Optuna') else 'best_epoch'
        if int(info.get(epoch_key, -1)) != spec['frozen_epoch']:
            raise ValueError(f'{spec["name"]}: best epoch is not the previously verified epoch')
        saved_base = info.get('model_initialization') if spec['name'].startswith('Optuna') else info.get('base_model')
        if saved_base != MODEL_ID:
            raise ValueError(f'{spec["name"]}: unexpected base-model identifier')
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f'The output directory already exists: {OUTPUT_DIR}\n'
            'This is a one-time test evaluation; nothing will be overwritten. '
            'Inspect the existing report rather than rerun or retune.'
        )


@torch.inference_mode()
def score_test(model, tokenizer, test_texts, device):
    dataset = TestTextDataset(test_texts, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collator)
    model.eval()
    chunks = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        probs = torch.softmax(model(**inputs).logits, dim=1)[:, 1]
        chunks.append(probs.cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all() or len(result) != len(test_texts):
        raise ValueError('Invalid model predictions')
    return result.astype(float)


def summarize(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold_frozen_on_validation': threshold,
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'failure_prevalence': float(y.mean()),
        'predicted_failure_fraction': float(pred.mean()),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }, pred


def main():
    print('=' * 78)
    print('FROZEN CHRONOLOGICAL TEST — THREE SAVED TRANSFORMER MODELS')
    print('=' * 78)
    print('No training, no Optuna, no new threshold search, no validation evaluation.')
    print('Metadata features: branch, is_pr, num_commits (timing provisional)')
    print('Comparison type: within-project chronological, NOT unseen-project')
    check_model_configs()      # Verify frozen settings BEFORE touching test labels.
    test = check_dataset()
    texts = test['metadata_text'].astype(str).tolist()
    y_test = test['label'].astype(int).to_numpy()
    print('Held-out chronological test builds:', len(test))
    print('Observed test failure fraction:', f'{y_test.mean():.4f}')

    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    all_rows = []
    predictions = test[['project_name', 'build_id', 'label']].reset_index().rename(
        columns={'index': 'sample_row_id', 'label': 'true_label'}
    )
    # Note: the trainable (fitted) classical baseline is not saved in the
    # supplied artifacts; do NOT silently retrain it in this evaluation.
    for spec in MODELS:
        print(f'\nLoading frozen {spec["name"]} ...', flush=True)
        checkpoint = spec['dir'] / spec['checkpoint']
        if spec['kind'] == 'full':
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
            model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        model = model.to(device)
        p = score_test(model, tokenizer, texts, device)
        metrics, binary = summarize(y_test, p, spec['frozen_threshold'])
        metrics['model'] = spec['name']
        metrics['test_builds'] = len(test)
        all_rows.append(metrics)
        prefix = ('distilbert' if spec['kind'] == 'full' else
                  'fixed_lora' if spec['name'].startswith('Fixed') else 'optuna_lora')
        predictions[f'{prefix}_probability'] = p
        predictions[f'{prefix}_predicted_failure'] = binary
        print(f'  F1={metrics["f1"]:.4f} | AP={metrics["average_precision"]:.4f} | '
              f'ROC-AUC={metrics["roc_auc"]:.4f} | FP={metrics["fp"]} | TN={metrics["tn"]}')
        del model, tokenizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    always_failure = np.ones_like(y_test)
    baseline_f1 = f1_score(y_test, always_failure, zero_division=0)
    print('\n' + '=' * 78)
    print('FROZEN TEST COMPARISON — CANDIDATE METADATA CLASSIFIERS')
    print('=' * 78)
    report = pd.DataFrame(all_rows)
    columns = ['model', 'threshold_frozen_on_validation', 'precision', 'recall',
               'f1', 'average_precision', 'roc_auc', 'predicted_failure_fraction',
               'tp', 'fp', 'tn', 'fn']
    with pd.option_context('display.max_columns', None, 'display.width', 180):
        print(report[columns].round(4).to_string(index=False))
    print(f'Always-failure F1 reference on these test labels: {baseline_f1:.4f}')
    print(f'Test failure prevalence (AP no-skill reference): {y_test.mean():.4f}')

    # New output only: never overwrite the training runs or a prior final test.
    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    report.to_csv(OUTPUT_DIR / 'frozen_three_transformer_test_metrics.csv', index=False)
    predictions.to_csv(OUTPUT_DIR / 'frozen_three_transformer_test_predictions.csv', index=False)
    summary = {
        'protocol': '5710_unique_builds_within_project_chronological_TEST_ONCE',
        'models': [row['model'] for row in all_rows],
        'features': EXPECTED_FEATURES,
        'metadata_not_CI_execution_logs': True,
        'historical_feature_availability_confirmed': False,
        'no_retraining_or_retuning': True,
        'test_builds': len(test),
        'always_failure_f1': float(baseline_f1),
        'test_failure_prevalence': float(y_test.mean()),
        'metrics': all_rows,
    }
    (OUTPUT_DIR / 'frozen_test_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSaved test metrics and predictions in:', OUTPUT_DIR)
    print('NO models retrained or tuned. Earlier result folders unchanged.')
    print('CAUTION: Test outcomes must NOT be used to revise model choice or thresholds.')
    print('CAUTION: Static metadata is NOT a running CI log; pre-build timing unverified.')


if __name__ == '__main__':
    main()

