import os
import json
import time
import random
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
# STEP 6
# BASELINE DISTILBERT + EFFICIENCY ANALYSIS
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

TRAIN_FILE = (
    BASE_DIR
    / "results_baseline"
    / "train_split.csv"
)

VAL_FILE = (
    BASE_DIR
    / "results_baseline"
    / "validation_split.csv"
)

TEST_FILE = (
    BASE_DIR
    / "results_baseline"
    / "test_split.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results_distilbert"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "best_distilbert_model"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

MODEL_DIR.mkdir(
    exist_ok=True
)

print("=" * 75)
print("STEP 6 - BASELINE DISTILBERT EFFICIENCY ANALYSIS")
print("=" * 75)


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

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "CUDA is not available. "
        "Training will use CPU."
    )


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )


# ============================================================
# LOAD EXACT SAME SPLITS AS BASELINES
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

TEXT_COLUMN = "log_100"

required_columns = [
    TEXT_COLUMN,
    "label"
]

for name, dataframe in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    missing = [
        col
        for col in required_columns
        if col not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{name} split missing columns: "
            + ", ".join(missing)
        )


for dataframe in [
    train_df,
    val_df,
    test_df
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


print("\nDataset split:")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

print(
    "\nTraining label distribution:"
)

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# TOKENIZER
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET CLASS
# ============================================================

class CICDDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.texts = (
            dataframe[TEXT_COLUMN]
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .tolist()
        )

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(
        self,
        index
    ):

        text = (
            self.texts[index]
        )

        label = (
            self.labels[index]
        )

        encoded = (
            self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        )

        item = {

            key:
                value.squeeze(0)

            for key, value
            in encoded.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )

        return item


train_dataset = CICDDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

val_dataset = CICDDataset(
    val_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = CICDDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading DistilBERT..."
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

trainable_percent = (
    trainable_params
    / total_params
    * 100
)

print("\n" + "=" * 75)
print("MODEL PARAMETERS")
print("=" * 75)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percent:.4f}%"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

number_classes = 2

total_training = (
    len(train_df)
)

class_weights = []

for class_id in range(
    number_classes
):

    class_count = (
        class_counts.get(
            class_id,
            1
        )
    )

    weight = (
        total_training
        /
        (
            number_classes
            * class_count
        )
    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
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


total_training_steps = (
    len(train_loader)
    * MAX_EPOCHS
)

warmup_steps = int(
    total_training_steps
    * 0.10
)


scheduler = (
    get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    labels_all = []
    probabilities_all = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            labels = (
                batch.pop("labels")
                .to(device)
            )

            inputs = {

                key:
                    value.to(device)

                for key, value
                in batch.items()
            }

            outputs = model(
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


    average_loss = (
        total_loss
        / max(
            len(loader),
            1
        )
    )

    return (
        np.array(
            labels_all
        ),

        np.array(
            probabilities_all
        ),

        average_loss
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1.0

    threshold_results = []

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

        threshold_results.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                "f1_score":
                    float(
                        score
                    )
            }
        )

        if score > best_f1:

            best_f1 = score

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
            threshold_results
        )
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 75)
print("TRAINING BASELINE DISTILBERT")
print("=" * 75)

best_validation_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0

training_history = []

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

    training_loss = 0.0

    for batch in train_loader:

        labels = (
            batch.pop("labels")
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

        logits = (
            outputs.logits
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        scheduler.step()

        training_loss += (
            loss.item()
        )


    average_train_loss = (
        training_loss
        / len(train_loader)
    )


    val_labels, \
    val_probabilities, \
    validation_loss = evaluate_model(
        model,
        val_loader
    )


    validation_threshold, \
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
            average_train_loss,
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
        "Best validation threshold:",
        round(
            validation_threshold,
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


    training_history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                average_train_loss,

            "validation_loss":
                validation_loss,

            "validation_f1":
                validation_f1,

            "validation_threshold":
                validation_threshold,

            "epoch_time_sec":
                epoch_time
        }
    )


    # ========================================================
    # SAVE BEST MODEL ACCORDING TO VALIDATION F1
    # ========================================================

    if validation_f1 > best_validation_f1:

        best_validation_f1 = (
            validation_f1
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        print(
            "New best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement."
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


total_training_time = (
    time.perf_counter()
    - training_start
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    OUTPUT_DIR
    / "03_distilbert_training_history.csv",
    index=False
)


# ============================================================
# RELOAD BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("RELOADING BEST DISTILBERT MODEL")
print("=" * 75)

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
final_validation_loss = evaluate_model(
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
    / "03_distilbert_threshold_search.csv",
    index=False
)


print(
    "\nSelected threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Best validation F1:",
    round(
        best_validation_f1,
        4
    )
)


# ============================================================
# TEST SET
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

test_labels, \
test_probabilities, \
test_loss = evaluate_model(
    model,
    test_loader
)


# ------------------------------------------------------------
# Default threshold = 0.50
# ------------------------------------------------------------

default_predictions = (
    test_probabilities
    >= 0.50
).astype(int)


default_f1 = f1_score(
    test_labels,
    default_predictions,
    zero_division=0
)


# ------------------------------------------------------------
# Validation-selected threshold
# ------------------------------------------------------------

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
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
    test_predictions
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
        f1,
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
    "Default threshold F1:",
    round(
        default_f1,
        4
    )
)

print(
    "Optimized threshold:",
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
# PARAMETER STORAGE SIZE
# ============================================================

parameter_bytes = sum(

    parameter.numel()
    * parameter.element_size()

    for parameter
    in model.parameters()
)

model_size_mb = (
    parameter_bytes
    / (1024 ** 2)
)


# ============================================================
# END-TO-END SINGLE BUILD LATENCY
#
# Includes tokenization + model inference.
# ============================================================

print("\n" + "=" * 75)
print("INFERENCE LATENCY")
print("=" * 75)

latency_texts = (
    test_df[TEXT_COLUMN]
    .head(
        min(
            100,
            len(test_df)
        )
    )
    .tolist()
)


# Warm-up
for text in latency_texts[:10]:

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    encoded = {

        key:
            value.to(device)

        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        _ = model(
            **encoded
        )


if device.type == "cuda":

    torch.cuda.synchronize()


latencies = []

model.eval()


for text in latency_texts:

    start = time.perf_counter()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    encoded = {

        key:
            value.to(device)

        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        _ = model(
            **encoded
        )

    if device.type == "cuda":

        torch.cuda.synchronize()

    end = time.perf_counter()

    latencies.append(
        (
            end - start
        )
        * 1000
    )


mean_latency = float(
    np.mean(
        latencies
    )
)

median_latency = float(
    np.median(
        latencies
    )
)

p95_latency = float(
    np.percentile(
        latencies,
        95
    )
)


print(
    "Mean latency:",
    round(
        mean_latency,
        3
    ),
    "ms/build"
)

print(
    "Median latency:",
    round(
        median_latency,
        3
    ),
    "ms/build"
)

print(
    "P95 latency:",
    round(
        p95_latency,
        3
    ),
    "ms/build"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "true_label":
            test_labels,

        "failure_probability":
            test_probabilities,

        "prediction":
            test_predictions
    }
)

prediction_df.to_csv(
    OUTPUT_DIR
    / "03_distilbert_test_predictions.csv",
    index=False
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

results = {

    "model":
        "Baseline_DistilBERT",

    "base_model":
        MODEL_NAME,

    "best_epoch":
        int(
            best_epoch
        ),

    "max_sequence_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "total_parameters":
        int(
            total_params
        ),

    "trainable_parameters":
        int(
            trainable_params
        ),

    "trainable_percentage":
        float(
            trainable_percent
        ),

    "parameter_size_mb":
        float(
            model_size_mb
        ),

    "training_time_sec":
        float(
            total_training_time
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
            f1
        ),

    "test_f1_default_threshold":
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

    "mean_latency_ms":
        mean_latency,

    "median_latency_ms":
        median_latency,

    "p95_latency_ms":
        p95_latency,

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
    / "03_distilbert_final_results.json",
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
    / "03_distilbert_final_results.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DISTILBERT FINAL SUMMARY")
print("=" * 75)

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
    "Optimized threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Test F1:",
    round(
        f1,
        4
    )
)

print(
    "Test ROC-AUC:",
    round(
        roc_auc,
        4
    )
)

print(
    "Test PR-AUC:",
    round(
        pr_auc,
        4
    )
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percent:.4f}%"
)

print(
    "Parameter size:",
    round(
        model_size_mb,
        2
    ),
    "MB"
)

print(
    "Training time:",
    round(
        total_training_time,
        2
    ),
    "seconds"
)

print(
    "Mean inference latency:",
    round(
        mean_latency,
        3
    ),
    "ms/build"
)

print(
    "P95 inference latency:",
    round(
        p95_latency,
        3
    ),
    "ms/build"
)

print(
    "\nResults folder:",
    OUTPUT_DIR
)

print(
    "\nSTEP 6 COMPLETED SUCCESSFULLY"
)