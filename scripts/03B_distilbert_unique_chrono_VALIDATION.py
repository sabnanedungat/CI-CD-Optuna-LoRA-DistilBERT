"""DistilBERT full-fine-tuning, VALIDATION ONLY, on 5,710 unique CI builds.

Input: static text of branch, is_pr, num_commits; NOT a CI execution log.
Historical pre-build feature availability remains provisional.

Uses within-project chronological train/validation split; no test inference.
Starts from CLEAN general pretrained distilbert-base-uncased, not any older
checkpoint fine-tuned on the previous 6,000-row dataset.

A new results folder protects existing experiment outputs.
"""
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
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
OUTPUT_DIR = ROOT / 'results_distilbert_unique_chrono_VALIDATION_ONLY'
CHECKPOINT_DIR = OUTPUT_DIR / 'best_model'

MODEL_ID = 'distilbert-base-uncased'  # NEVER load the old results_distilbert checkpoint
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 5
PATIENCE = 2


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
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # dynamic padding per batch, more efficient for short metadata
        )
        self.labels = frame['label'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        result = {key: value[idx] for key, value in self.encodings.items()}
        result['labels'] = self.labels[idx]
        return result


@torch.no_grad()
def validation_probabilities(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        true = batch.pop('labels')
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs).logits
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(true.cpu().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def main():
    print('=' * 73)
    print('CLEAN DISTILBERT — 5,710 UNIQUE BUILDS, CHRONOLOGICAL VALIDATION ONLY')
    print('=' * 73)
    data = check_inputs()
    train = data.loc[data['split'].eq('train')].copy()
    val = data.loc[data['split'].eq('validation')].copy()
    test_count = int(data['split'].eq('test').sum())
    if (len(train), len(val), test_count) != (3651, 915, 1144):
        raise ValueError('Unexpected chronological split sizes')
    if train['label'].nunique() != 2 or val['label'].nunique() != 2:
        raise ValueError('Training and validation must each have both classes')
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f'{OUTPUT_DIR} already contains files. Move/rename it before rerunning '
            'to avoid overwriting an earlier experiment.'
        )

    print('Input features: branch, is_pr, num_commits (static metadata)')
    print('Model initialization:', MODEL_ID, '(fresh classification head)')
    print('Training builds:', len(train))
    print('Validation builds:', len(val))
    print('Test builds NOT evaluated:', test_count)
    print('Validation failure prevalence:', f"{val['label'].mean():.4f}")
    print('Checkpoint selection: validation average precision (PR-AUC/AP)')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    # Downloads general pretrained weights if not already cached; does NOT use
    # the existing task-fine-tuned checkpoint or any of its model weights.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model.to(device)

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
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.10 * total_steps),
        num_training_steps=total_steps,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False) if not OUTPUT_DIR.exists() else None
    best_ap, best_epoch, stale = -1.0, -1, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            targets = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.item())

        yval, probs = validation_probabilities(model, val_loader, device)
        ap = float(average_precision_score(yval, probs))
        roc = float(roc_auc_score(yval, probs))
        train_loss = loss_total / len(train_loader)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'validation_ap': ap, 'validation_roc_auc': roc})
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, validation_AP={ap:.4f}, validation_ROC_AUC={roc:.4f}')

        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            print('  Saved new best validation-AP checkpoint.')
        else:
            stale += 1
            if stale >= PATIENCE:
                print('  Stopped after two epochs without validation-AP improvement.')
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    # Reload checkpoint selected by validation. Test data is never tokenized
    # or passed to a model in this script.
    best_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    labels, probs = validation_probabilities(best_model, val_loader, device)
    candidate_thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    choices = [(float(f1_score(labels, probs >= t, zero_division=0)), float(t))
               for t in candidate_thresholds]
    selected_f1, threshold = max(choices, key=lambda pair: (pair[0], -pair[1]))
    predicted = (probs >= threshold).astype(int)
    result = {
        'dataset': '5710_unique_builds_within_project_chronological',
        'features': ['branch', 'is_pr', 'num_commits'],
        'input_type': 'static_metadata_text_not_execution_log',
        'feature_timing_confirmed': False,
        'base_model': MODEL_ID,
        'checkpoint_selected_by': 'validation_average_precision',
        'best_epoch': best_epoch,
        'train_builds': len(train), 'validation_builds': len(val),
        'test_builds_not_evaluated': test_count,
        'validation_threshold_selected_by_F1': threshold,
        'validation_f1_selected_optimistic': selected_f1,
        'validation_precision': float(precision_score(labels, predicted, zero_division=0)),
        'validation_recall': float(recall_score(labels, predicted, zero_division=0)),
        'validation_average_precision': float(average_precision_score(labels, probs)),
        'validation_roc_auc': float(roc_auc_score(labels, probs)),
        'validation_predicted_failure_fraction': float(predicted.mean()),
        'validation_failure_prevalence': float(labels.mean()),
        'training_elapsed_seconds': round(time.perf_counter() - started, 2),
    }
    (OUTPUT_DIR / 'validation_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('\n' + '=' * 73)
    print('VALIDATION RESULTS — NOT FINAL TEST / NOT PUBLICATION SCORES')
    print('=' * 73)
    for key in ('best_epoch', 'validation_threshold_selected_by_F1',
                'validation_precision', 'validation_recall',
                'validation_f1_selected_optimistic', 'validation_average_precision',
                'validation_roc_auc', 'validation_predicted_failure_fraction'):
        print(f'{key}: {result[key]}')
    print('Validation AP no-skill reference (failure prevalence):',
          f"{result['validation_failure_prevalence']:.4f}")
    print('Results directory:', OUTPUT_DIR)
    print('TEST SET NOT EVALUATED. No old checkpoint or results directory modified.')
    print('CAUTION: historical pre-build feature availability is still provisional.')
    print('CAUTION: static metadata text is NOT a partial CI log.')


if __name__ == '__main__':
    main()

