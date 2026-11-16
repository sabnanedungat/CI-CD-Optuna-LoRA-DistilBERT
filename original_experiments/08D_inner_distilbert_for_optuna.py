import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# STEP 13B1
# INNER DISTILBERT BASE MODEL FOR NESTED OPTUNA
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 3

BASE_DIR = Path(__file__).resolve().parent

NESTED_DIR = (
    BASE_DIR
    / "results_cross_project"
    / "nested_optuna"
)

TRAIN_FILE = (
    NESTED_DIR
    / "inner_train.csv"
)

VAL_FILE = (
    NESTED_DIR
    / "inner_optuna_validation.csv"
)

OUTPUT_DIR = (
    NESTED_DIR
    / "inner_distilbert"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "best_inner_distilbert"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 90)
print("STEP 13B1 - INNER DISTILBERT BASE FOR OPTUNA")
print("=" * 90)

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device
)

# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    TRAIN_FILE,
    VAL_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{file_path}"
        )

# ============================================================
# LOAD INNER DATA ONLY
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False
)

TEXT_COLUMN = "log_text"

for dataframe in [
    train_df,
    val_df
]:

    dataframe[TEXT_COLUMN] = (
        dataframe[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

print(
    "\nInner training rows:",
    len(train_df)
)

print(
    "Inner training projects:",
    train_df[
        "project_name"
    ].nunique()
)

print(
    "\nInner Optuna-validation rows:",
    len(val_df)
)

print(
    "Inner Optuna-validation projects:",
    val_df[
        "project_name"
    ].nunique()
)

# ============================================================
# VERIFY PROJECT SEPARATION
# ============================================================

train_projects = set(
    train_df[
        "project_name"
    ].unique()
)

val_projects = set(
    val_df[
        "project_name"
    ].unique()
)

overlap = (
    train_projects
    & val_projects
)

if overlap:

    raise RuntimeError(
        f"Project leakage detected: {overlap}"
    )

print(
    "\nInner project overlap:",
    len(overlap)
)

print(
    "Project separation check: PASSED"
)

# ============================================================
# TOKENIZER
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)

# ============================================================
# DATASET CLASS
# ============================================================

class CICDDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.texts = (
            dataframe[
                TEXT_COLUMN
            ]
            .tolist()
        )

        self.labels = (
            dataframe[
                "label"
            ]
            .tolist()
        )

    def __len__(
        self
    ):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index
    ):

        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {

            key:
                value.squeeze(0)

            for key, value
            in encoded.items()
        }

        item[
            "labels"
        ] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item

# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    CICDDataset(
        train_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    CICDDataset(
        val_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# FRESH DISTILBERT
# ============================================================

print(
    "\nLoading fresh DistilBERT..."
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
)

model.to(
    device
)

# ============================================================
# PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df[
        "label"
    ]
    .value_counts()
    .sort_index()
)

weights = []

for class_id in [
    0,
    1
]:

    count = class_counts.get(
        class_id,
        1
    )

    weight = (
        len(train_df)
        /
        (
            2
            * count
        )
    )

    weights.append(
        weight
    )

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print(
    "\nClass weights:",
    class_weights
    .detach()
    .cpu()
    .numpy()
)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = (
    len(train_loader)
    * MAX_EPOCHS
)

scheduler = (
    get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps
            * 0.10
        ),
        num_training_steps=total_steps
    )
)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    current_model,
    loader
):

    current_model.eval()

    labels_all = []
    probabilities_all = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            labels = (
                batch.pop(
                    "labels"
                )
                .to(device)
            )

            inputs = {

                key:
                    value.to(device)

                for key, value
                in batch.items()
            }

            outputs = current_model(
                **inputs
            )

            logits = (
                outputs.logits
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )[:, 1]
            )

            labels_all.extend(
                labels
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            probabilities_all.extend(
                probabilities
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.array(
            labels_all
        ),

        np.array(
            probabilities_all
        ),

        total_loss
        / max(
            len(loader),
            1
        )
    )

# ============================================================
# TRAIN
#
# IMPORTANT:
# Epoch selection uses F1 at fixed threshold 0.50.
#
# Threshold is NOT optimized here because this validation set
# will later be used specifically for Optuna tuning.
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "TRAINING INNER DISTILBERT"
)

print(
    "=" * 90
)

best_validation_f1 = -1.0
best_epoch = 0

history = []

training_start = (
    time.perf_counter()
)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    epoch_start = (
        time.perf_counter()
    )

    model.train()

    total_training_loss = 0.0

    for batch in train_loader:

        labels = (
            batch.pop(
                "labels"
            )
            .to(device)
        )

        inputs = {

            key:
                value.to(device)

            for key, value
            in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            **inputs
        )

        loss = criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        total_training_loss += (
            loss.item()
        )

    average_training_loss = (
        total_training_loss
        / len(train_loader)
    )

    # ========================================================
    # INNER OPTUNA-VALIDATION PROJECTS
    # ========================================================

    val_labels, \
    val_probabilities, \
    validation_loss = evaluate_model(
        model,
        val_loader
    )

    val_predictions = (
        val_probabilities
        >= 0.50
    ).astype(int)

    validation_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    validation_precision = precision_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    validation_recall = recall_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    validation_f1 = f1_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    validation_roc_auc = roc_auc_score(
        val_labels,
        val_probabilities
    )

    validation_pr_auc = average_precision_score(
        val_labels,
        val_probabilities
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        f"\nEpoch {epoch}/{MAX_EPOCHS}"
    )

    print(
        "Training loss:",
        round(
            average_training_loss,
            4
        )
    )

    print(
        "Validation loss:",
        round(
            validation_loss,
            4
        )
    )

    print(
        "Validation accuracy:",
        round(
            validation_accuracy,
            4
        )
    )

    print(
        "Validation precision:",
        round(
            validation_precision,
            4
        )
    )

    print(
        "Validation recall:",
        round(
            validation_recall,
            4
        )
    )

    print(
        "Validation F1 @ 0.50:",
        round(
            validation_f1,
            4
        )
    )

    print(
        "Validation ROC-AUC:",
        round(
            validation_roc_auc,
            4
        )
    )

    print(
        "Validation PR-AUC:",
        round(
            validation_pr_auc,
            4
        )
    )

    print(
        "Epoch time:",
        round(
            epoch_time,
            2
        ),
        "seconds"
    )

    history.append(
        {
            "epoch":
                epoch,

            "training_loss":
                average_training_loss,

            "validation_loss":
                validation_loss,

            "validation_accuracy":
                validation_accuracy,

            "validation_precision":
                validation_precision,

            "validation_recall":
                validation_recall,

            "validation_f1_at_0.50":
                validation_f1,

            "validation_roc_auc":
                validation_roc_auc,

            "validation_pr_auc":
                validation_pr_auc,

            "epoch_time_sec":
                epoch_time
        }
    )

    # ========================================================
    # SELECT BEST INNER BASE USING FIXED-THRESHOLD F1
    # ========================================================

    if (
        validation_f1
        > best_validation_f1
    ):

        best_validation_f1 = (
            validation_f1
        )

        best_epoch = (
            epoch
        )

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        print(
            "New best INNER DistilBERT saved."
        )

    else:

        print(
            "No F1 improvement."
        )

# ============================================================
# TRAINING TIME
# ============================================================

training_time = (
    time.perf_counter()
    - training_start
)

# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_DIR
    / "08D_inner_distilbert_history.csv",
    index=False
)

# ============================================================
# RELOAD BEST MODEL FOR FINAL INNER VALIDATION REPORT
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "RELOADING BEST INNER DISTILBERT"
)

print(
    "=" * 90
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

model.to(
    device
)

val_labels, \
val_probabilities, \
validation_loss = evaluate_model(
    model,
    val_loader
)

val_predictions = (
    val_probabilities
    >= 0.50
).astype(int)

final_accuracy = accuracy_score(
    val_labels,
    val_predictions
)

final_precision = precision_score(
    val_labels,
    val_predictions,
    zero_division=0
)

final_recall = recall_score(
    val_labels,
    val_predictions,
    zero_division=0
)

final_f1 = f1_score(
    val_labels,
    val_predictions,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    val_labels,
    val_probabilities
)

final_pr_auc = average_precision_score(
    val_labels,
    val_probabilities
)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model":
        "Inner_DistilBERT_for_Optuna",

    "inner_training_projects":
        int(
            train_df[
                "project_name"
            ].nunique()
        ),

    "inner_validation_projects":
        int(
            val_df[
                "project_name"
            ].nunique()
        ),

    "inner_training_rows":
        int(
            len(train_df)
        ),

    "inner_validation_rows":
        int(
            len(val_df)
        ),

    "best_epoch":
        int(
            best_epoch
        ),

    "validation_threshold":
        0.50,

    "validation_accuracy":
        float(
            final_accuracy
        ),

    "validation_precision":
        float(
            final_precision
        ),

    "validation_recall":
        float(
            final_recall
        ),

    "validation_f1":
        float(
            final_f1
        ),

    "validation_roc_auc":
        float(
            final_roc_auc
        ),

    "validation_pr_auc":
        float(
            final_pr_auc
        ),

    "training_time_sec":
        float(
            training_time
        ),

    "total_parameters":
        int(
            total_parameters
        ),

    "trainable_parameters":
        int(
            trainable_parameters
        ),

    "device":
        str(
            device
        )
}

with open(
    OUTPUT_DIR
    / "08D_inner_distilbert_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

pd.DataFrame(
    [results]
).to_csv(
    OUTPUT_DIR
    / "08D_inner_distilbert_results.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "INNER DISTILBERT FINAL SUMMARY"
)

print(
    "=" * 90
)

print(
    "Inner training projects:",
    train_df[
        "project_name"
    ].nunique()
)

print(
    "Inner validation projects:",
    val_df[
        "project_name"
    ].nunique()
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Validation accuracy:",
    round(
        final_accuracy,
        4
    )
)

print(
    "Validation precision:",
    round(
        final_precision,
        4
    )
)

print(
    "Validation recall:",
    round(
        final_recall,
        4
    )
)

print(
    "Validation F1 @ 0.50:",
    round(
        final_f1,
        4
    )
)

print(
    "Validation ROC-AUC:",
    round(
        final_roc_auc,
        4
    )
)

print(
    "Validation PR-AUC:",
    round(
        final_pr_auc,
        4
    )
)

print(
    "Training time:",
    round(
        training_time,
        2
    ),
    "seconds"
)

print(
    "\nInner base model:"
)

print(
    MODEL_DIR
)

print(
    "\nOfficial validation projects were NOT used."
)

print(
    "Official test projects were NOT used."
)

print(
    "\nSTEP 13B1 COMPLETED SUCCESSFULLY"
)