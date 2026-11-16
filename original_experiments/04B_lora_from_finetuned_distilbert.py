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
# STEP 7B
# LoRA APPLIED TO THE ALREADY FINE-TUNED DISTILBERT
# ============================================================

SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16

# Fixed LoRA configuration
# Still NO Optuna in this experiment
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 4
PATIENCE = 2

BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_DIR = (
    BASE_DIR
    / "results_distilbert"
    / "best_distilbert_model"
)

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
    / "results_lora_finetuned_base"
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
print("STEP 7B - LoRA FROM FINE-TUNED DISTILBERT")
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

# ============================================================
# CHECK BASELINE MODEL
# ============================================================

if not BASE_MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Fine-tuned DistilBERT model not found:\n{BASE_MODEL_DIR}"
    )

print(
    "\nFine-tuned DistilBERT source:"
)

print(
    BASE_MODEL_DIR
)

# ============================================================
# LOAD SAME SPLITS
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

for df in [
    train_df,
    val_df,
    test_df
]:

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    df["label"] = (
        df["label"]
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

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        BASE_MODEL_DIR
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
            dataframe[TEXT_COLUMN]
            .tolist()
        )

        self.labels = (
            dataframe["label"]
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
            key: value.squeeze(0)

            for key, value
            in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


train_loader = DataLoader(
    CICDDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    CICDDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    CICDDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# LOAD ALREADY FINE-TUNED DISTILBERT
# ============================================================

print(
    "\nLoading previously fine-tuned DistilBERT..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)

# ============================================================
# APPLY LoRA
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

print("\n" + "=" * 78)
print("PARAMETER EFFICIENCY")
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
    "Trainable percentage:",
    f"{trainable_percentage:.4f}%"
)

model.print_trainable_parameters()

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
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

    weights.append(

        len(train_df)
        /
        (
            2
            * count
        )
    )

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

criterion = (
    torch.nn.CrossEntropyLoss(
        weight=class_weights
    )
)

# ============================================================
# OPTIMIZER
# ============================================================

trainable_model_parameters = [

    p

    for p in model.parameters()

    if p.requires_grad
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

scheduler = (
    get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=int(
            total_steps * 0.10
        ),

        num_training_steps=total_steps
    )
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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
                .cpu()
                .numpy()
                .tolist()
            )

            probabilities_all.extend(
                probabilities
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
        / len(loader)
    )

# ============================================================
# THRESHOLD
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

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
# TRAIN
# ============================================================

print("\n" + "=" * 78)
print("TRAINING LoRA ON FINE-TUNED DISTILBERT")
print("=" * 78)

best_validation_f1 = -1
best_epoch = 0

no_improvement = 0

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
            trainable_model_parameters,
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

    val_labels, \
    val_probs, \
    val_loss = evaluate(
        model,
        val_loader
    )

    threshold, \
    val_f1, \
    _ = find_best_threshold(
        val_labels,
        val_probs
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
        "Threshold:",
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
                train_loss,

            "validation_loss":
                val_loss,

            "threshold":
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

        no_improvement = 0

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        print(
            "New best adapter saved."
        )

    else:

        no_improvement += 1

        print(
            "No improvement."
        )

    if no_improvement >= PATIENCE:

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
    / "04B_training_history.csv",

    index=False
)

# ============================================================
# RELOAD BEST ADAPTER
# ============================================================

print("\n" + "=" * 78)
print("RELOADING BEST MODEL")
print("=" * 78)

reload_base = (

    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)

model = PeftModel.from_pretrained(

    reload_base,

    MODEL_DIR
)

model.to(
    device
)

# ============================================================
# VALIDATION THRESHOLD
# ============================================================

val_labels, \
val_probs, \
_ = evaluate(
    model,
    val_loader
)

best_threshold, \
best_validation_f1, \
threshold_df = find_best_threshold(
    val_labels,
    val_probs
)

threshold_df.to_csv(

    OUTPUT_DIR
    / "04B_threshold_search.csv",

    index=False
)

# ============================================================
# TEST
# ============================================================

test_labels, \
test_probs, \
_ = evaluate(
    model,
    test_loader
)

predictions = (
    test_probs
    >= best_threshold
).astype(int)

default_predictions = (
    test_probs
    >= 0.50
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

default_f1 = f1_score(
    test_labels,
    default_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    test_labels,
    test_probs
)

pr_auc = average_precision_score(
    test_labels,
    test_probs
)

cm = confusion_matrix(
    test_labels,
    predictions
)

# ============================================================
# ADAPTER SIZE
# ============================================================

adapter_bytes = sum(

    f.stat().st_size

    for f in MODEL_DIR.rglob("*")

    if f.is_file()
)

adapter_size_mb = (
    adapter_bytes
    / (1024 ** 2)
)

# ============================================================
# LATENCY
# ============================================================

latency_texts = (
    test_df[TEXT_COLUMN]
    .head(100)
    .tolist()
)

model.eval()

# Warm-up
for text in latency_texts[:10]:

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    encoded = {

        key:
            value.to(device)

        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        model(
            **encoded
        )

latencies = []

for text in latency_texts:

    start = (
        time.perf_counter()
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    encoded = {

        key:
            value.to(device)

        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        model(
            **encoded
        )

    latencies.append(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

mean_latency = float(
    np.mean(
        latencies
    )
)

p95_latency = float(
    np.percentile(
        latencies,
        95
    )
)

# ============================================================
# RESULTS
# ============================================================

results = {

    "model":
        "LoRA_from_finetuned_DistilBERT",

    "lora_rank":
        LORA_R,

    "lora_alpha":
        LORA_ALPHA,

    "lora_dropout":
        LORA_DROPOUT,

    "learning_rate":
        LEARNING_RATE,

    "best_epoch":
        best_epoch,

    "total_parameters":
        total_params,

    "trainable_parameters":
        trainable_params,

    "trainable_percentage":
        trainable_percentage,

    "training_time_sec":
        training_time,

    "adapter_size_mb":
        adapter_size_mb,

    "validation_f1":
        best_validation_f1,

    "threshold":
        best_threshold,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "test_f1":
        test_f1,

    "test_f1_default_threshold":
        default_f1,

    "roc_auc":
        roc_auc,

    "pr_auc":
        pr_auc,

    "mean_latency_ms":
        mean_latency,

    "p95_latency_ms":
        p95_latency,

    "tn":
        int(cm[0][0]),

    "fp":
        int(cm[0][1]),

    "fn":
        int(cm[1][0]),

    "tp":
        int(cm[1][1])
}

with open(

    OUTPUT_DIR
    / "04B_final_results.json",

    "w"

) as f:

    json.dump(
        results,
        f,
        indent=4
    )

pd.DataFrame(
    [results]
).to_csv(

    OUTPUT_DIR
    / "04B_final_results.csv",

    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("CORRECTED LoRA FINAL SUMMARY")
print("=" * 78)

print(
    "Base model:",
    BASE_MODEL_DIR
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
    "Test accuracy:",
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
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percentage:.4f}%"
)

print(
    "Adapter size:",
    round(
        adapter_size_mb,
        2
    ),
    "MB"
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
    "Mean latency:",
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
    "\nSTEP 7B COMPLETED SUCCESSFULLY"
)