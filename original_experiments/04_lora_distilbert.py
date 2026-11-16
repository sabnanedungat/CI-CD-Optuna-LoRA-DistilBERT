import json
import os
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

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel
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
# STEP 7
# FIXED-CONFIGURATION LoRA-DistilBERT
# NO OPTUNA
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16

# ------------------------------------------------------------
# Fixed LoRA configuration
# These values are NOT selected by Optuna.
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10

LEARNING_RATE = 1e-4
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
    / "results_lora"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "best_lora_adapter"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

MODEL_DIR.mkdir(
    exist_ok=True
)

print("=" * 78)
print("STEP 7 - LoRA-DistilBERT WITHOUT OPTUNA")
print("=" * 78)

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
        "CUDA unavailable - CPU training."
    )

# ============================================================
# LOAD SAME SPLITS
# ============================================================

for path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


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

# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

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


    def __len__(self):

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

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


train_dataset = CICDDataset(
    train_df
)

val_dataset = CICDDataset(
    val_df
)

test_dataset = CICDDataset(
    test_df
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# BASE MODEL
# ============================================================

print(
    "\nLoading base DistilBERT..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
)

# ============================================================
# LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(

    task_type=TaskType.SEQ_CLS,

    inference_mode=False,

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_lin",
        "v_lin"
    ],

    modules_to_save=[
        "pre_classifier",
        "classifier"
    ]
)

model = get_peft_model(
    base_model,
    lora_config
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

trainable_percentage = (
    trainable_params
    / total_params
    * 100
)

frozen_params = (
    total_params
    - trainable_params
)

print("\n" + "=" * 78)
print("LoRA PARAMETER EFFICIENCY")
print("=" * 78)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Frozen parameters:",
    f"{frozen_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percentage:.4f}%"
)

print("\nPEFT report:")

model.print_trainable_parameters()

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

total_training = len(
    train_df
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
        total_training
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
# Only trainable parameters are supplied.
# ============================================================

trainable_model_parameters = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
]


optimizer = AdamW(

    trainable_model_parameters,

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


total_steps = (

    len(train_loader)
    * MAX_EPOCHS
)

warmup_steps = int(
    total_steps
    * 0.10
)


scheduler = (
    get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=warmup_steps,

        num_training_steps=total_steps
    )
)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
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


    average_loss = (

        total_loss
        / max(
            len(loader),
            1
        )
    )

    return (

        np.array(
            all_labels
        ),

        np.array(
            all_probabilities
        ),

        average_loss
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

                "f1_score":
                    float(
                        score
                    )
            }
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold


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

print("\n" + "=" * 78)
print("TRAINING LoRA-DistilBERT")
print("=" * 78)

history = []

best_validation_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0

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

    total_train_loss = 0.0

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

        logits = (
            outputs.logits
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_model_parameters,
            max_norm=1.0
        )

        optimizer.step()

        scheduler.step()

        total_train_loss += (
            loss.item()
        )


    average_train_loss = (

        total_train_loss
        / len(train_loader)
    )


    val_labels, \
    val_probabilities, \
    val_loss = evaluate(
        model,
        val_loader
    )


    threshold, \
    val_f1, \
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
            val_f1,
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
                average_train_loss,

            "validation_loss":
                val_loss,

            "validation_threshold":
                threshold,

            "validation_f1":
                val_f1,

            "epoch_time_sec":
                epoch_time
        }
    )


    if val_f1 > best_validation_f1:

        best_validation_f1 = (
            val_f1
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
            "New best LoRA adapter saved."
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
            "\nEarly stopping triggered."
        )

        break


total_training_time = (

    time.perf_counter()
    - training_start
)

# ============================================================
# SAVE HISTORY
# ============================================================

pd.DataFrame(
    history
).to_csv(

    OUTPUT_DIR
    / "04_lora_training_history.csv",

    index=False
)

# ============================================================
# ADAPTER CHECKPOINT SIZE
# ============================================================

adapter_size_bytes = 0

for file in MODEL_DIR.rglob(
    "*"
):

    if file.is_file():

        adapter_size_bytes += (
            file.stat().st_size
        )


adapter_size_mb = (

    adapter_size_bytes
    / (1024 ** 2)
)

# ============================================================
# RELOAD BEST LoRA MODEL
# ============================================================

print("\n" + "=" * 78)
print("RELOADING BEST LoRA MODEL")
print("=" * 78)


reload_base_model = (

    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
)


model = PeftModel.from_pretrained(

    reload_base_model,

    MODEL_DIR
)

model.to(
    device
)

# ============================================================
# FINAL VALIDATION THRESHOLD
# ============================================================

val_labels, \
val_probabilities, \
final_validation_loss = evaluate(

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
    / "04_lora_threshold_search.csv",

    index=False
)


print(
    "\nBest validation threshold:",
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
# TEST
# ============================================================

print("\n" + "=" * 78)
print("FINAL LoRA TEST EVALUATION")
print("=" * 78)


test_labels, \
test_probabilities, \
test_loss = evaluate(

    model,

    test_loader
)

# ------------------------------------------------------------
# Default 0.50 threshold
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

predictions = (

    test_probabilities
    >= best_threshold

).astype(int)


accuracy = accuracy_score(
    test_labels,
    predictions
)

precision = precision_score(
    test_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    predictions,
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
    predictions
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
    "F1 at default 0.50:",
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
    "Threshold:",
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
# INFERENCE LATENCY
# ============================================================

print("\n" + "=" * 78)
print("LoRA INFERENCE LATENCY")
print("=" * 78)


latency_texts = (

    test_df[
        TEXT_COLUMN
    ]
    .head(
        min(
            100,
            len(test_df)
        )
    )
    .tolist()
)

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

model.eval()

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


for text in latency_texts:

    start = (
        time.perf_counter()
    )

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

    end = (
        time.perf_counter()
    )

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
    "Mean:",
    round(
        mean_latency,
        3
    ),
    "ms/build"
)

print(
    "Median:",
    round(
        median_latency,
        3
    ),
    "ms/build"
)

print(
    "P95:",
    round(
        p95_latency,
        3
    ),
    "ms/build"
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "true_label":
            test_labels,

        "failure_probability":
            test_probabilities,

        "prediction":
            predictions
    }
)


prediction_df.to_csv(

    OUTPUT_DIR
    / "04_lora_test_predictions.csv",

    index=False
)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model":
        "LoRA_DistilBERT_No_Optuna",

    "base_model":
        MODEL_NAME,

    "lora_rank":
        LORA_R,

    "lora_alpha":
        LORA_ALPHA,

    "lora_dropout":
        LORA_DROPOUT,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "best_epoch":
        int(
            best_epoch
        ),

    "total_parameters":
        int(
            total_params
        ),

    "trainable_parameters":
        int(
            trainable_params
        ),

    "frozen_parameters":
        int(
            frozen_params
        ),

    "trainable_percentage":
        float(
            trainable_percentage
        ),

    "adapter_checkpoint_size_mb":
        float(
            adapter_size_mb
        ),

    "training_time_sec":
        float(
            total_training_time
        ),

    "validation_f1":
        float(
            best_validation_f1
        ),

    "optimized_threshold":
        float(
            best_threshold
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
    / "04_lora_final_results.json",

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
    / "04_lora_final_results.csv",

    index=False
)

# ============================================================
# DISTILBERT VS LoRA COMPARISON
# ============================================================

distilbert_result_file = (

    BASE_DIR
    / "results_distilbert"
    / "03_distilbert_final_results.json"
)


if distilbert_result_file.exists():

    with open(
        distilbert_result_file,
        "r"
    ) as file:

        distilbert_results = (
            json.load(
                file
            )
        )


    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Baseline DistilBERT",

                "Trainable Parameters":
                    distilbert_results[
                        "trainable_parameters"
                    ],

                "Trainable %":
                    distilbert_results[
                        "trainable_percentage"
                    ],

                "Training Time Sec":
                    distilbert_results[
                        "training_time_sec"
                    ],

                "F1":
                    distilbert_results[
                        "test_f1"
                    ],

                "F1 Default Threshold":
                    distilbert_results[
                        "test_f1_default_threshold"
                    ],

                "ROC-AUC":
                    distilbert_results[
                        "test_roc_auc"
                    ],

                "PR-AUC":
                    distilbert_results[
                        "test_pr_auc"
                    ],

                "Mean Latency ms":
                    distilbert_results[
                        "mean_latency_ms"
                    ]
            },

            {
                "Model":
                    "LoRA-DistilBERT",

                "Trainable Parameters":
                    trainable_params,

                "Trainable %":
                    trainable_percentage,

                "Training Time Sec":
                    total_training_time,

                "F1":
                    test_f1,

                "F1 Default Threshold":
                    default_f1,

                "ROC-AUC":
                    roc_auc,

                "PR-AUC":
                    pr_auc,

                "Mean Latency ms":
                    mean_latency
            }
        ]
    )


    comparison.to_csv(

        OUTPUT_DIR
        / "04_distilbert_vs_lora.csv",

        index=False
    )


    print("\n" + "=" * 78)
    print("DISTILBERT VS LoRA")
    print("=" * 78)

    print(
        comparison.to_string(
            index=False
        )
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("LoRA-DISTILBERT FINAL SUMMARY")
print("=" * 78)

print(
    "LoRA rank:",
    LORA_R
)

print(
    "LoRA alpha:",
    LORA_ALPHA
)

print(
    "LoRA dropout:",
    LORA_DROPOUT
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
    "Test F1:",
    round(
        test_f1,
        4
    )
)

print(
    "Test F1 at 0.50:",
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
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percentage:.4f}%"
)

print(
    "Adapter checkpoint size:",
    round(
        adapter_size_mb,
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
    "P95 latency:",
    round(
        p95_latency,
        3
    ),
    "ms/build"
)

print(
    "\nResults:",
    OUTPUT_DIR
)

print(
    "\nSTEP 7 COMPLETED SUCCESSFULLY"
)