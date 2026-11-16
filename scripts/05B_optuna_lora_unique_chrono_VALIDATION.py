"""Validation-only Optuna-LoRA DistilBERT pilot, on 5,710 unique CI builds.

Same static metadata text and within-project chronological train/validation split
as 03B/04C. Never evaluates the held-out test partition. The metadata is not
an execution log, and prediction-time feature availability is not proven.

Search is deliberately small for a CPU pilot; it is NOT an exhaustive HPO study.
Run from the folder containing the two candidate CSVs below.
"""

from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import optuna
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / 'unique_builds_three_feature_METADATA_TEXT_CANDIDATE.csv'
SPLIT_CSV = ROOT / 'unique_builds_chronological_split_manifest.csv'
OUTPUT_DIR = ROOT / 'results_optuna_lora_unique_chrono_VALIDATION_ONLY'
CHECKPOINT_DIR = OUTPUT_DIR / 'best_adapter'

MODEL_ID = 'distilbert-base-uncased'
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
N_TRIALS = 4                    # Small CPU-friendly PILOT, not exhaustive HPO.
TRIAL_EPOCHS = 2                # Cheap exploration; final model uses up to 5 epochs.
FINAL_MAX_EPOCHS = 5
FINAL_PATIENCE = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_check_data():
    for filename in (INPUT_CSV, SPLIT_CSV):
        if not filename.is_file():
            raise FileNotFoundError(f'Missing required file: {filename}')
    data = pd.read_csv(INPUT_CSV, dtype={'build_id': 'string'}, low_memory=False)
    manifest = pd.read_csv(SPLIT_CSV, dtype={'build_id': 'string'}, low_memory=False)
    required = {'sample_row_id', 'project_name', 'build_id', 'label', 'split'}
    if (required | {'metadata_text'}) - set(data.columns):
        raise ValueError('Metadata CSV missing required columns')
    if required - set(manifest.columns):
        raise ValueError('Manifest CSV missing required columns')
    if len(data) != 5710 or len(manifest) != 5710:
        raise ValueError('Expected exactly 5,710 rows in each file')
    if data['sample_row_id'].duplicated().any() or manifest['sample_row_id'].duplicated().any():
        raise ValueError('Duplicate sample_row_id')
    if data.duplicated(['project_name', 'build_id']).any():
        raise ValueError('Repeated CI build')
    data = data.set_index('sample_row_id').reindex(range(5710))
    manifest = manifest.set_index('sample_row_id').reindex(range(5710))
    for field in ('project_name', 'build_id', 'label', 'split'):
        if data[field].isna().any() or manifest[field].isna().any():
            raise ValueError(f'Null or misaligned {field}')
        if not np.array_equal(data[field].astype(str).to_numpy(), manifest[field].astype(str).to_numpy()):
            raise ValueError(f'Manifest/data disagreement for {field}')
    if data['metadata_text'].isna().any() or data['metadata_text'].astype(str).str.strip().eq('').any():
        raise ValueError('Empty metadata text')
    if not data['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown split')
    if not data['label'].isin([0, 1]).all():
        raise ValueError('Unexpected label')
    train = data.loc[data['split'].eq('train')].reset_index(drop=True)
    validation = data.loc[data['split'].eq('validation')].reset_index(drop=True)
    # Do not select test rows or access their text or labels.
    test_count = int(data['split'].eq('test').sum())
    if (len(train), len(validation), test_count) != (3651, 915, 1144):
        raise ValueError('Unexpected chronological partition counts')
    if train['label'].nunique() != 2 or validation['label'].nunique() != 2:
        raise ValueError('Both classes must be present in train and validation')
    return train, validation, test_count


class MetadataDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.encoding = tokenizer(frame['metadata_text'].astype(str).tolist(),
                                  truncation=True, max_length=MAX_LENGTH, padding=False)
        self.y = frame['label'].astype(int).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encoding.items()}
        item['labels'] = self.y[index]
        return item


def make_loaders(train_data, val_data, tokenizer):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED),
                              collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collator, num_workers=0)
    return train_loader, val_loader


def create_model(params, device):
    # Always start afresh from GENERAL pretrained weights.
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    adapter = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=int(params['rank']), lora_alpha=int(params['alpha']),
        lora_dropout=float(params['dropout']), bias='none',
        target_modules=['q_lin', 'v_lin'],
        modules_to_save=['pre_classifier', 'classifier'],
    )
    return get_peft_model(base, adapter).to(device)


@torch.no_grad()
def val_predictions(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        y = batch.pop('labels')
        x = {key: value.to(device) for key, value in batch.items()}
        prob = torch.softmax(model(**x).logits, dim=1)[:, 1]
        labels.extend(y.cpu().tolist())
        probabilities.extend(prob.cpu().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def fit_epochs(model, loaders, device, params, weights, max_epochs, patience=None,
               checkpoint=None, trial=None):
    train_loader, val_loader = loaders
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('No trainable parameters')
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(trainable, lr=float(params['learning_rate']),
                      weight_decay=float(params['weight_decay']))
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)
    best_ap, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            target = batch.pop('labels').to(device)
            x = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**x).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        yval, prob = val_predictions(model, val_loader, device)
        ap = float(average_precision_score(yval, prob))
        roc = float(roc_auc_score(yval, prob))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'validation_ap': ap, 'validation_roc_auc': roc})
        print(f'  Epoch {epoch}: train_loss={np.mean(losses):.4f}, '
              f'validation_AP={ap:.4f}, validation_ROC_AUC={roc:.4f}', flush=True)
        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            if checkpoint is not None:
                model.save_pretrained(checkpoint)
        else:
            stale += 1
        if trial is not None:
            trial.report(ap, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience is not None and stale >= patience:
            print('  Early stopping after no validation-AP improvement.', flush=True)
            break
    return best_ap, best_epoch, history


def main():
    print('=' * 74)
    print('OPTUNA-LoRA PILOT — UNIQUE-BUILD CHRONOLOGICAL VALIDATION ONLY')
    print('=' * 74)
    train, validation, test_count = load_and_check_data()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f'Existing results directory is not empty: {OUTPUT_DIR}')
    print('Features: branch, is_pr, num_commits (static metadata; timing provisional)')
    print('Train builds:', len(train), '| Validation builds:', len(validation),
          '| Test builds NOT evaluated:', test_count)
    print('Validation failure prevalence:', f"{validation['label'].mean():.4f}")
    print('Trials:', N_TRIALS, '| Epochs per trial:', TRIAL_EPOCHS,
          '| Final max epochs:', FINAL_MAX_EPOCHS)
    print('Objective: validation average precision (AP), NOT test scores')
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_dataset = MetadataDataset(train, tokenizer)
    val_dataset = MetadataDataset(validation, tokenizer)
    labels = train['label'].value_counts().reindex([0, 1], fill_value=0)
    weights = torch.tensor([len(train) / (2 * labels[0]),
                            len(train) / (2 * labels[1])],
                           dtype=torch.float32, device=device)

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 2e-5, 2e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 0.05, log=True),
            'rank': trial.suggest_categorical('rank', [4, 8, 16]),
            'alpha': trial.suggest_categorical('alpha', [8, 16, 32]),
            'dropout': trial.suggest_categorical('dropout', [0.0, 0.1, 0.2]),
        }
        print(f'\nTrial {trial.number + 1}/{N_TRIALS}: {params}', flush=True)
        seed_everything(SEED)  # Consistent starting initialization across trials.
        model = create_model(params, device)
        loaders = make_loaders(train_dataset, val_dataset, tokenizer)
        try:
            best_ap, _, _ = fit_epochs(model, loaders, device, params, weights,
                                       TRIAL_EPOCHS, trial=trial)
            return best_ap
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.NopPruner())
    search_start = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.perf_counter() - search_start
    print('\nBest Optuna validation AP:', f'{study.best_value:.4f}')
    print('Best search params:', study.best_params)

    # Final training from GENERAL pretrained weights, not any old task checkpoint.
    print('\nFinal training with selected hyperparameters (VALIDATION ONLY)', flush=True)
    seed_everything(SEED)
    best_params = study.best_params
    best_model = create_model(best_params, device)
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    loaders = make_loaders(train_dataset, val_dataset, tokenizer)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_start = time.perf_counter()
    best_ap, best_epoch, history = fit_epochs(
        best_model, loaders, device, best_params, weights, FINAL_MAX_EPOCHS,
        patience=FINAL_PATIENCE, checkpoint=CHECKPOINT_DIR)
    training_seconds = time.perf_counter() - training_start
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    del best_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore the selected adapter on a freshly loaded general-pretraining base.
    seed_everything(SEED)
    fresh_base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model = PeftModel.from_pretrained(fresh_base, CHECKPOINT_DIR).to(device)
    yval, prob = val_predictions(model, loaders[1], device)
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    choices = [(float(f1_score(yval, prob >= t, zero_division=0)), float(t)) for t in thresholds]
    selected_f1, threshold = max(choices, key=lambda pair: (pair[0], -pair[1]))
    pred = (prob >= threshold).astype(int)
    result = {
        'dataset': '5710_unique_builds_within_project_chronological',
        'features': ['branch', 'is_pr', 'num_commits'],
        'input_type': 'static_metadata_text_not_execution_log',
        'historical_feature_timing_proven': False,
        'method': 'optuna_lora_small_pilot',
        'model_initialization': MODEL_ID,
        'trials': N_TRIALS,
        'trial_epochs': TRIAL_EPOCHS,
        'optuna_best_trial_validation_ap': float(study.best_value),
        'best_params': best_params,
        'best_final_epoch': int(best_epoch),
        'trainable_parameters': int(trainable_params),
        'total_parameters': int(total_params),
        'trainable_percentage': float(100 * trainable_params / total_params),
        'search_seconds': float(search_seconds),
        'final_training_seconds': float(training_seconds),
        'train_builds': len(train),
        'validation_builds': len(validation),
        'test_builds_not_evaluated': test_count,
        'validation_selected_threshold': threshold,
        'validation_f1_selected_optimistic': selected_f1,
        'validation_precision': float(precision_score(yval, pred, zero_division=0)),
        'validation_recall': float(recall_score(yval, pred, zero_division=0)),
        'validation_ap': float(average_precision_score(yval, prob)),
        'validation_roc_auc': float(roc_auc_score(yval, prob)),
        'validation_predicted_failure_fraction': float(pred.mean()),
        'validation_failure_prevalence': float(yval.mean()),
    }
    (OUTPUT_DIR / 'validation_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    study.trials_dataframe().to_csv(OUTPUT_DIR / 'optuna_trials.csv', index=False)
    print('\n' + '=' * 74)
    print('OPTUNA-LoRA VALIDATION RESULTS — NO HELD-OUT TEST')
    print('=' * 74)
    for key in ('trials', 'optuna_best_trial_validation_ap', 'best_final_epoch',
                'validation_selected_threshold', 'validation_precision',
                'validation_recall', 'validation_f1_selected_optimistic',
                'validation_ap', 'validation_roc_auc',
                'validation_predicted_failure_fraction'):
        print(f'{key}: {result[key]}')
    print('Results directory:', OUTPUT_DIR)
    print('TEST SET NOT EVALUATED. NO TEST-BASED HYPERPARAMETER OR THRESHOLD SEARCH.')
    print('CAUTION: Short metadata is NOT CI execution logs; feature timing unverified.')


if __name__ == '__main__':
    main()

