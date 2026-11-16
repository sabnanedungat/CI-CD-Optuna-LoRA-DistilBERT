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
    average_precision_score,
    confusion_matrix
)

# ============================================================
# STEP 12
# CROSS-PROJECT DISTILBERT BASELINE
#
# Train projects      = 14
# Validation projects = 3
# Test projects       = 3 completely unseen projects
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 5
PATIENCE = 2

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "results_cross_project"
)

TRAIN_FILE = (
    DATA_DIR
    / "cross_project_train.csv"
)

VAL_FILE = (
    DATA_DIR
    / "cross_project_validation.csv"
)

TEST_FILE = (
    DATA_DIR
    / "cross_project_test.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "distilbert"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "best_cross_project_distilbert"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 85)
print("STEP 12 - CROSS-PROJECT DISTILBERT")
print("=" * 85)

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

for path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

TEXT_COLUMN = "log_text"

for name, dataframe in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
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
        f"\n{name}:",
        len(dataframe),
        "rows |",
        dataframe[
            "project_name"
        ].nunique(),
        "projects"
    )

# ============================================================
# VERIFY PROJECT SEPARATION AGAIN
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

test_projects = set(
    test_df[
        "project_name"
    ].unique()
)

assert len(
    train_projects
    & val_projects
) == 0

assert len(
    train_projects
    & test_projects
) == 0

assert len(
    val_projects
    & test_projects
) == 0

print(
    "\nProject overlap verification: PASSED"
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
# DATASET
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
# LOADERS
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

test_loader = DataLoader(
    CICDDataset(
        test_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# MODEL
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
# PARAMETER COUNTS
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTotal parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
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

    count = (
        class_counts.get(
            class_id,
            1
        )
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

criterion = (
    torch.nn.CrossEntropyLoss(
        weight=class_weights
    )
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
# EVALUATION
# ============================================================

def evaluate_model(
    current_model,
    loader
):

    current_model.eval()

    all_labels = []
    all_probabilities = []

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

            outputs = (
                current_model(
                    **inputs
                )
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

            all_labels.extend(
                labels
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.array(
            all_labels
        ),

        np.array(
            all_probabilities
        ),

        total_loss
        / max(
            len(loader),
            1
        )
    )

# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1.0

    rows = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.01
    ):

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        rows.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                "f1":
                    float(
                        score
                    )
            }
        )

        if score > best_f1:

            best_f1 = (
                score
            )

            best_threshold = (
                threshold
            )

    return (
        float(
            best_threshold
        ),

        float(
            best_f1
        ),

        pd.DataFrame(
            rows
        )
    )

# ============================================================
# TRAINING
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "CROSS-PROJECT DISTILBERT TRAINING"
)

print(
    "=" * 85
)

best_validation_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0

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

    train_loss = 0.0

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

        train_loss += (
            loss.item()
        )

    train_loss /= (
        len(train_loader)
    )

    # --------------------------------------------------------
    # VALIDATION PROJECTS ONLY
    # --------------------------------------------------------

    val_labels, \
    val_probabilities, \
    val_loss = evaluate_model(
        model,
        val_loader
    )

    threshold, \
    validation_f1, \
    _ = find_best_threshold(
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
        "Train loss:",
        round(
            train_loss,
            4
        )
    )

    print(
        "Validation loss:",
        round(
            val_loss,
            4
        )
    )

    print(
        "Validation threshold:",
        round(
            threshold,
            2
        )
    )

    print(
        "Validation F1:",
        round(
            validation_f1,
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

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss,

            "validation_threshold":
                threshold,

            "validation_f1":
                validation_f1,

            "epoch_time_sec":
                epoch_time
        }
    )

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

        epochs_without_improvement = 0

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        print(
            "New best cross-project DistilBERT saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement."
        )

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break

training_time = (
    time.perf_counter()
    - training_start
)

pd.DataFrame(
    history
).to_csv(
    OUTPUT_DIR
    / "08B_training_history.csv",
    index=False
)

# ============================================================
# RELOAD BEST MODEL
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "RELOADING BEST CROSS-PROJECT MODEL"
)

print(
    "=" * 85
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

# ============================================================
# FINAL VALIDATION THRESHOLD
# ============================================================

val_labels, \
val_probabilities, \
_ = evaluate_model(
    model,
    val_loader
)

best_threshold, \
best_validation_f1, \
threshold_df = find_best_threshold(
    val_labels,
    val_probabilities
)

threshold_df.to_csv(
    OUTPUT_DIR
    / "08B_threshold_search.csv",
    index=False
)

print(
    "\nFinal validation threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Validation F1:",
    round(
        best_validation_f1,
        4
    )
)

# ============================================================
# UNSEEN TEST PROJECTS
#
# Test projects have never been used for:
# training
# validation
# threshold selection
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "UNSEEN PROJECT TEST EVALUATION"
)

print(
    "=" * 85
)

test_labels, \
test_probabilities, \
_ = evaluate_model(
    model,
    test_loader
)

default_predictions = (
    test_probabilities
    >= 0.50
).astype(int)

optimized_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)

default_f1 = f1_score(
    test_labels,
    default_predictions,
    zero_division=0
)

accuracy = accuracy_score(
    test_labels,
    optimized_predictions
)

precision = precision_score(
    test_labels,
    optimized_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    optimized_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    optimized_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    test_labels,
    test_probabilities
)

pr_auc = average_precision_score(
    test_labels,
    test_probabilities
)

cm = confusion_matrix(
    test_labels,
    optimized_predictions
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1:",
    round(
        test_f1,
        4
    )
)

print(
    "F1 @ 0.50:",
    round(
        default_f1,
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc,
        4
    )
)

print(
    "PR-AUC:",
    round(
        pr_auc,
        4
    )
)

print(
    "Selected threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "\nConfusion matrix:"
)

print(
    cm
)

# ============================================================
# PROJECT-WISE TEST PERFORMANCE
# ============================================================

project_rows = []

test_output = test_df[
    [
        "project_name",
        "label"
    ]
].copy()

test_output[
    "probability"
] = test_probabilities

test_output[
    "prediction"
] = optimized_predictions

for project, group in test_output.groupby(
    "project_name"
):

    project_labels = (
        group[
            "label"
        ]
        .to_numpy()
    )

    project_predictions = (
        group[
            "prediction"
        ]
        .to_numpy()
    )

    project_probabilities = (
        group[
            "probability"
        ]
        .to_numpy()
    )

    row = {
        "project":
            project,

        "n_builds":
            len(group),

        "failure_rate":
            float(
                project_labels.mean()
            ),

        "accuracy":
            accuracy_score(
                project_labels,
                project_predictions
            ),

        "precision":
            precision_score(
                project_labels,
                project_predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                project_labels,
                project_predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                project_labels,
                project_predictions,
                zero_division=0
            )
    }

    if len(
        np.unique(
            project_labels
        )
    ) == 2:

        row[
            "roc_auc"
        ] = roc_auc_score(
            project_labels,
            project_probabilities
        )

        row[
            "pr_auc"
        ] = average_precision_score(
            project_labels,
            project_probabilities
        )

    else:

        row[
            "roc_auc"
        ] = np.nan

        row[
            "pr_auc"
        ] = np.nan

    project_rows.append(
        row
    )

project_df = pd.DataFrame(
    project_rows
)

print(
    "\n"
    + "=" * 85
)

print(
    "TEST PERFORMANCE BY UNSEEN PROJECT"
)

print(
    "=" * 85
)

print(
    project_df.round(
        4
    ).to_string(
        index=False
    )
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_output.to_csv(
    OUTPUT_DIR
    / "08B_cross_project_test_predictions.csv",
    index=False
)

project_df.to_csv(
    OUTPUT_DIR
    / "08B_unseen_project_metrics.csv",
    index=False
)

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results = {

    "model":
        "Cross_Project_DistilBERT",

    "training_projects":
        len(
            train_projects
        ),

    "validation_projects":
        len(
            val_projects
        ),

    "test_projects":
        len(
            test_projects
        ),

    "training_rows":
        len(
            train_df
        ),

    "validation_rows":
        len(
            val_df
        ),

    "test_rows":
        len(
            test_df
        ),

    "best_epoch":
        int(
            best_epoch
        ),

    "optimized_threshold":
        float(
            best_threshold
        ),

    "validation_f1":
        float(
            best_validation_f1
        ),

    "test_accuracy":
        float(
            accuracy
        ),

    "test_precision":
        float(
            precision
        ),

    "test_recall":
        float(
            recall
        ),

    "test_f1":
        float(
            test_f1
        ),

    "test_f1_at_0.50":
        float(
            default_f1
        ),

    "test_roc_auc":
        float(
            roc_auc
        ),

    "test_pr_auc":
        float(
            pr_auc
        ),

    "training_time_sec":
        float(
            training_time
        ),

    "total_parameters":
        int(
            total_params
        ),

    "trainable_parameters":
        int(
            trainable_params
        ),

    "tn":
        int(
            cm[0][0]
        ),

    "fp":
        int(
            cm[0][1]
        ),

    "fn":
        int(
            cm[1][0]
        ),

    "tp":
        int(
            cm[1][1]
        ),

    "device":
        str(
            device
        )
}

with open(
    OUTPUT_DIR
    / "08B_cross_project_distilbert_results.json",
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
    / "08B_cross_project_distilbert_results.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "CROSS-PROJECT DISTILBERT FINAL SUMMARY"
)

print(
    "=" * 85
)

print(
    "Training projects:",
    len(
        train_projects
    )
)

print(
    "Validation projects:",
    len(
        val_projects
    )
)

print(
    "Unseen test projects:",
    len(
        test_projects
    )
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Validation F1:",
    round(
        best_validation_f1,
        4
    )
)

print(
    "Threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Unseen-project test F1:",
    round(
        test_f1,
        4
    )
)

print(
    "Unseen-project ROC-AUC:",
    round(
        roc_auc,
        4
    )
)

print(
    "Unseen-project PR-AUC:",
    round(
        pr_auc,
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
    "\nModel saved at:"
)

print(
    MODEL_DIR
)

print(
    "\nSTEP 12 COMPLETED SUCCESSFULLY"
)