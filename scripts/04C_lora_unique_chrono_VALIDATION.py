"""Clean LoRA-DistilBERT, validation ONLY, on 5,710 distinct CI builds.

Uses the identical static metadata text and chronological train/validation
partitions as 03B_distilbert_unique_chrono_VALIDATION.py.

Starting weights: Hugging Face 'distilbert-base-uncased', NOT the previous
6,000-row task-fine-tuned checkpoint and NOT the new 03B checkpoint.

Candidate features: branch, is_pr, num_commits. Their historical availability
at build trigger time is not fully proven. NOT progressive CI execution logs.

Does NOT read labels or text from held-out test builds for prediction.
Does NOT run Optuna or overwrite prior experiment outputs.
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
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
OUTPUT_DIR = ROOT / 'results_lora_unique_chrono_VALIDATION_ONLY'
CHECKPOINT_DIR = OUTPUT_DIR / 'best_adapter'

MODEL_ID = 'distilbert-base-uncased'
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
MAX_EPOCHS = 5
PATIENCE = 2
# Fixed pilot hyperparameters, selected without inspecting any test outcome.
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10


def check_inputs():
    for path in (INPUT_CSV, SPLIT_CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input file: {path}')

    data = pd.read_csv(INPUT_CSV, dtype={'build_id': 'string'}, low_memory=False)
    manifest = pd.read_csv(SPLIT_CSV, dtype={'build_id': 'string'}, low_memory=False)
    needed = {'sample_row_id', 'project_name', 'build_id', 'split', 'label', 'metadata_text'}
    if missing := needed - set(data.columns):
        raise ValueError(f'Metadata CSV missing columns: {sorted(missing)}')
    if missing := (needed - {'metadata_text'}) - set(manifest.columns):
        raise ValueError(f'Manifest CSV missing columns: {sorted(missing)}')
    if len(data) != 5710 or len(manifest) != 5710:
        raise ValueError(f'Expected 5710 rows, got {len(data)} and {len(manifest)}')
    if data['sample_row_id'].duplicated().any() or manifest['sample_row_id'].duplicated().any():
        raise ValueError('Duplicate sample_row_id')
    if data.duplicated(['project_name', 'build_id']).any():
        raise ValueError('Duplicate project/build_id in metadata CSV')

    data = data.set_index('sample_row_id').reindex(range(5710))
    manifest = manifest.set_index('sample_row_id').reindex(range(5710))
    for name in ('project_name', 'build_id', 'split', 'label'):
        if data[name].isna().any() or manifest[name].isna().any():
            raise ValueError(f'Null or missing alignment data: {name}')
        if not np.array_equal(data[name].astype(str).to_numpy(), manifest[name].astype(str).to_numpy()):
            raise ValueError(f'Metadata / manifest mismatch: {name}')
    if data['metadata_text'].isna().any() or data['metadata_text'].astype(str).str.strip().eq('').any():
        raise ValueError('Empty metadata text')
    if not data['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unexpected split labels')
    if not data['label'].isin([0, 1]).all():
        raise ValueError('Unexpected classification labels')
    if data['project_name'].nunique() != 20:
        raise ValueError('Expected 20 projects')
    return data.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.encodings = tokenizer(
            frame['metadata_text'].astype(str).tolist(),
            truncation=True, max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = frame['label'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item


@torch.no_grad()
def validation_probabilities(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        true = batch.pop('labels')
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(true.cpu().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def main():
    print('=' * 73)
    print('CLEAN LoRA-DISTILBERT — UNIQUE-BUILD CHRONOLOGICAL VALIDATION ONLY')
    print('=' * 73)
    data = check_inputs()
    train = data.loc[data['split'].eq('train')].copy()
    val = data.loc[data['split'].eq('validation')].copy()
    # Count only; do not access test text/labels for any prediction or tuning.
    test_count = int(data['split'].eq('test').sum())
    if (len(train), len(val), test_count) != (3651, 915, 1144):
        raise ValueError('Unexpected chronological split sizes')
    if train['label'].nunique() != 2 or val['label'].nunique() != 2:
        raise ValueError('Both training and validation must contain both classes')
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f'{OUTPUT_DIR} already contains results; will not overwrite them')

    print('Features: branch, is_pr, num_commits (static metadata; timing provisional)')
    print('Fresh starting model:', MODEL_ID)
    print('LoRA settings:', {'rank': LORA_RANK, 'alpha': LORA_ALPHA, 'dropout': LORA_DROPOUT,
                             'learning_rate': LEARNING_RATE})
    print('Train builds:', len(train))
    print('Validation builds:', len(val))
    print('Test builds NOT evaluated:', test_count)
    print('Validation failure prevalence:', f"{val['label'].mean():.4f}")
    print('Checkpoint selection metric: validation AP')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    # Both newly initialized DistilBERT classification layers MUST be saved and
    # trained alongside the adapter for a valid adapter-only checkpoint.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        target_modules=['q_lin', 'v_lin'],
        modules_to_save=['pre_classifier', 'classifier'],
    )
    model = get_peft_model(base, lora_config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError('No trainable adapter/classifier parameters')
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in trainable)
    print('Trainable parameters:', f'{trainable_params:,}', 'of', f'{total_params:,}',
          f'({100 * trainable_params / total_params:.3f}%)')

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    train_loader = DataLoader(
        TextDataset(train, tokenizer), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        TextDataset(val, tokenizer), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collator,
    )

    counts = train['label'].value_counts().reindex([0, 1], fill_value=0)
    class_weights = torch.tensor(
        [len(train) / (2 * counts[0]), len(train) / (2 * counts[1])],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.10 * steps), num_training_steps=steps,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False) if not OUTPUT_DIR.exists() else None
    best_ap, best_epoch, stale = -1.0, -1, 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        for batch in train_loader:
            targets = batch.pop('labels').to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss_total += float(loss.item())

        yval, prob = validation_probabilities(model, val_loader, device)
        ap = float(average_precision_score(yval, prob))
        roc_auc = float(roc_auc_score(yval, prob))
        history.append({'epoch': epoch, 'train_loss': train_loss_total / len(train_loader),
                        'validation_ap': ap, 'validation_roc_auc': roc_auc})
        print(f'Epoch {epoch}: train_loss={train_loss_total / len(train_loader):.4f}, '
              f'validation_AP={ap:.4f}, validation_ROC_AUC={roc_auc:.4f}')
        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            print('  Saved new best validation-AP adapter.')
        else:
            stale += 1
            if stale >= PATIENCE:
                print('  Stopped after two epochs without validation-AP improvement.')
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    # Restore from general pretrained weights + the validation-selected adapter.
    # Do not reload the old task-fine-tuned model or the new full-FT model.
    fresh_base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    best_model = PeftModel.from_pretrained(fresh_base, CHECKPOINT_DIR).to(device)
    labels, probs = validation_probabilities(best_model, val_loader, device)
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    choices = [(float(f1_score(labels, probs >= threshold, zero_division=0)), float(threshold))
               for threshold in candidates]
    best_f1, selected_threshold = max(choices, key=lambda pair: (pair[0], -pair[1]))
    predictions = (probs >= selected_threshold).astype(int)

    summary = {
        'dataset': '5710_unique_builds_within_project_chronological',
        'features': ['branch', 'is_pr', 'num_commits'],
        'input_type': 'static_metadata_text_not_execution_log',
        'feature_timing_confirmed': False,
        'base_model': MODEL_ID,
        'method': 'lora_fixed_configuration',
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'trainable_percentage': float(100 * trainable_params / total_params),
        'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT, 'learning_rate': LEARNING_RATE,
        'best_epoch': best_epoch,
        'checkpoint_selected_by': 'validation_average_precision',
        'train_builds': len(train), 'validation_builds': len(val),
        'test_builds_not_evaluated': test_count,
        'validation_threshold_selected_by_F1': selected_threshold,
        'validation_f1_selected_optimistic': best_f1,
        'validation_precision': float(precision_score(labels, predictions, zero_division=0)),
        'validation_recall': float(recall_score(labels, predictions, zero_division=0)),
        'validation_average_precision': float(average_precision_score(labels, probs)),
        'validation_roc_auc': float(roc_auc_score(labels, probs)),
        'validation_predicted_failure_fraction': float(predictions.mean()),
        'validation_failure_prevalence': float(labels.mean()),
        'training_elapsed_seconds': float(time.perf_counter() - start),
    }
    (OUTPUT_DIR / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n' + '=' * 73)
    print('LORA VALIDATION RESULTS — NOT FINAL TEST / NOT PUBLICATION SCORES')
    print('=' * 73)
    for key in (
        'best_epoch', 'validation_threshold_selected_by_F1',
        'validation_precision', 'validation_recall',
        'validation_f1_selected_optimistic', 'validation_average_precision',
        'validation_roc_auc', 'validation_predicted_failure_fraction',
    ):
        print(f'{key}: {summary[key]}')
    print('Validation AP no-skill reference:', f"{summary['validation_failure_prevalence']:.4f}")
    print('Results directory:', OUTPUT_DIR)
    print('TEST SET NOT EVALUATED. Previous experiments preserved.')
    print('CAUTION: historical pre-build feature availability remains provisional.')
    print('CAUTION: static metadata is NOT a progressive CI execution log.')


if __name__ == '__main__':
    main()

