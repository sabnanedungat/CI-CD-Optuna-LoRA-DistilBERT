import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

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
# STEP 8
# OPTUNA-OPTIMIZED LoRA-DistilBERT
# ============================================================

SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16

# ------------------------------------------------------------
# Optuna settings
# ------------------------------------------------------------

N_TRIALS = 10

# Shorter training during hyperparameter search.
TRIAL_EPOCHS = 2

# Final best model gets fuller training.
FINAL_MAX_EPOCHS = 4
FINAL_PATIENCE = 2


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
    / "results_optuna_lora"
)

BEST_MODEL_DIR = (
    OUTPUT_DIR
    / "best_optuna_lora_adapter"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

BEST_MODEL_DIR.mkdir(
    exist_ok=True
)

print("=" * 80)
print("STEP 8 - OPTUNA-LoRA-DistilBERT")
print("=" * 80)

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

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
else:
    print(
        "CPU training mode"
    )

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not BASE_MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Fine-tuned DistilBERT not found:\n"
        f"{BASE_MODEL_DIR}"
    )

for file_path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:
    if not file_path.exists():
        raise FileNotFoundError(
            str(file_path)
        )

# ============================================================
# DATA
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


print("\nDataset:")

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
# DATASET CLASS
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

# ============================================================
# DATA LOADERS
# ============================================================

def create_loaders():

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
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

    return (
        train_loader,
        val_loader,
        test_loader
    )

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

class_weights = []

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

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

# ============================================================
# CREATE LoRA MODEL
# ============================================================

def create_lora_model(
    rank,
    alpha,
    dropout
):

    base_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL_DIR
        )
    )

    lora_config = LoraConfig(

        task_type=TaskType.SEQ_CLS,

        inference_mode=False,

        r=rank,

        lora_alpha=alpha,

        lora_dropout=dropout,

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

    return model

# ============================================================
# EVALUATION
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
# OPTUNA OBJECTIVE
#
# IMPORTANT:
# Threshold remains fixed at 0.50 during Optuna tuning.
# Threshold optimization happens only AFTER best
# hyperparameters are selected.
# ============================================================

def objective(
    trial
):

    learning_rate = trial.suggest_float(
        "learning_rate",
        2e-5,
        2e-4,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-5,
        0.05,
        log=True
    )

    lora_rank = trial.suggest_categorical(
        "lora_rank",
        [
            4,
            8,
            16
        ]
    )

    lora_alpha = trial.suggest_categorical(
        "lora_alpha",
        [
            8,
            16,
            32,
            64
        ]
    )

    lora_dropout = trial.suggest_float(
        "lora_dropout",
        0.0,
        0.20
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"OPTUNA TRIAL "
        f"{trial.number}"
    )

    print(
        "=" * 80
    )

    print(
        "Learning rate:",
        learning_rate
    )

    print(
        "Weight decay:",
        weight_decay
    )

    print(
        "LoRA rank:",
        lora_rank
    )

    print(
        "LoRA alpha:",
        lora_alpha
    )

    print(
        "LoRA dropout:",
        lora_dropout
    )


    train_loader, \
    val_loader, \
    _ = create_loaders()


    model = create_lora_model(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout
    )


    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay
    )


    total_steps = (
        len(train_loader)
        * TRIAL_EPOCHS
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


    best_trial_f1 = -1.0


    for epoch in range(
        1,
        TRIAL_EPOCHS + 1
    ):

        model.train()

        training_loss = 0.0

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
                trainable_parameters,
                1.0
            )

            optimizer.step()

            scheduler.step()

            training_loss += (
                loss.item()
            )


        val_labels, \
        val_probabilities, \
        val_loss = evaluate_model(
            model,
            val_loader
        )


        val_predictions = (
            val_probabilities
            >= 0.50
        ).astype(int)


        val_f1 = f1_score(
            val_labels,
            val_predictions,
            zero_division=0
        )


        best_trial_f1 = max(
            best_trial_f1,
            val_f1
        )


        print(
            f"Epoch {epoch}: "
            f"Train Loss="
            f"{training_loss / len(train_loader):.4f}, "
            f"Val Loss="
            f"{val_loss:.4f}, "
            f"Val F1@0.50="
            f"{val_f1:.4f}"
        )


        trial.report(
            val_f1,
            step=epoch
        )


        if trial.should_prune():

            print(
                "Trial pruned."
            )

            del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise optuna.TrialPruned()


    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return float(
        best_trial_f1
    )

# ============================================================
# OPTUNA STUDY
# ============================================================

print("\n" + "=" * 80)
print("STARTING OPTUNA SEARCH")
print("=" * 80)

print(
    "Number of trials:",
    N_TRIALS
)

print(
    "Epochs per trial:",
    TRIAL_EPOCHS
)


sampler = optuna.samplers.TPESampler(
    seed=SEED
)


pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,
    n_warmup_steps=1
)


study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    study_name="cicd_optuna_lora"
)


optuna_start = (
    time.perf_counter()
)


study.optimize(
    objective,
    n_trials=N_TRIALS
)


optuna_search_time = (
    time.perf_counter()
    - optuna_start
)

# ============================================================
# SAVE OPTUNA RESULTS
# ============================================================

study.trials_dataframe().to_csv(
    OUTPUT_DIR
    / "05_optuna_all_trials.csv",
    index=False
)


best_params = (
    study.best_params
)


with open(
    OUTPUT_DIR
    / "05_optuna_best_parameters.json",
    "w"
) as file:

    json.dump(
        {
            "best_validation_f1_at_0.50":
                study.best_value,

            "best_trial":
                study.best_trial.number,

            "search_time_sec":
                optuna_search_time,

            **best_params
        },
        file,
        indent=4
    )


print("\n" + "=" * 80)
print("OPTUNA BEST RESULT")
print("=" * 80)

print(
    "Best trial:",
    study.best_trial.number
)

print(
    "Best validation F1 @ 0.50:",
    round(
        study.best_value,
        4
    )
)

print(
    "\nBest parameters:"
)

for key, value in best_params.items():

    print(
        key,
        "=",
        value
    )

# ============================================================
# FINAL TRAINING USING BEST PARAMETERS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TRAINING WITH BEST OPTUNA PARAMETERS")
print("=" * 80)


train_loader, \
val_loader, \
test_loader = create_loaders()


model = create_lora_model(

    rank=best_params[
        "lora_rank"
    ],

    alpha=best_params[
        "lora_alpha"
    ],

    dropout=best_params[
        "lora_dropout"
    ]
)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


trainable_percentage = (
    trainable_parameters_count
    / total_parameters
    * 100
)


trainable_parameters = [
    p
    for p in model.parameters()
    if p.requires_grad
]


optimizer = AdamW(

    trainable_parameters,

    lr=best_params[
        "learning_rate"
    ],

    weight_decay=best_params[
        "weight_decay"
    ]
)


total_steps = (
    len(train_loader)
    * FINAL_MAX_EPOCHS
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


best_validation_f1_05 = -1.0

best_epoch = 0

no_improvement = 0

history = []


final_training_start = (
    time.perf_counter()
)


for epoch in range(
    1,
    FINAL_MAX_EPOCHS + 1
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

        loss = criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            1.0
        )

        optimizer.step()

        scheduler.step()

        total_train_loss += (
            loss.item()
        )


    val_labels, \
    val_probabilities, \
    val_loss = evaluate_model(
        model,
        val_loader
    )


    val_predictions_05 = (
        val_probabilities
        >= 0.50
    ).astype(int)


    val_f1_05 = f1_score(
        val_labels,
        val_predictions_05,
        zero_division=0
    )


    epoch_time = (
        time.perf_counter()
        - epoch_start
    )


    print(
        f"\nEpoch {epoch}/"
        f"{FINAL_MAX_EPOCHS}"
    )

    print(
        "Train loss:",
        round(
            total_train_loss
            / len(train_loader),
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
        "Validation F1 @ 0.50:",
        round(
            val_f1_05,
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
                total_train_loss
                / len(train_loader),

            "validation_loss":
                val_loss,

            "validation_f1_at_0.50":
                val_f1_05,

            "epoch_time_sec":
                epoch_time
        }
    )


    if (
        val_f1_05
        > best_validation_f1_05
    ):

        best_validation_f1_05 = (
            val_f1_05
        )

        best_epoch = epoch

        no_improvement = 0

        model.save_pretrained(
            BEST_MODEL_DIR
        )

        tokenizer.save_pretrained(
            BEST_MODEL_DIR
        )

        print(
            "New best Optuna-LoRA model saved."
        )

    else:

        no_improvement += 1

        print(
            "No improvement."
        )


    if (
        no_improvement
        >= FINAL_PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


final_training_time = (
    time.perf_counter()
    - final_training_start
)


pd.DataFrame(
    history
).to_csv(
    OUTPUT_DIR
    / "05_final_training_history.csv",
    index=False
)

# ============================================================
# ADAPTER CHECKPOINT SIZE
# ============================================================

adapter_bytes = sum(

    file.stat().st_size

    for file in BEST_MODEL_DIR.rglob("*")

    if file.is_file()
)


adapter_size_mb = (
    adapter_bytes
    / (1024 ** 2)
)

# ============================================================
# RELOAD BEST MODEL
# ============================================================

reload_base = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)


model = PeftModel.from_pretrained(
    reload_base,
    BEST_MODEL_DIR
)


model.to(
    device
)

# ============================================================
# NOW OPTIMIZE THRESHOLD
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION-BASED THRESHOLD OPTIMIZATION")
print("=" * 80)


val_labels, \
val_probabilities, \
_ = evaluate_model(
    model,
    val_loader
)


best_threshold, \
optimized_validation_f1, \
threshold_df = find_best_threshold(
    val_labels,
    val_probabilities
)


threshold_df.to_csv(
    OUTPUT_DIR
    / "05_threshold_search.csv",
    index=False
)


print(
    "Optimized threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Optimized validation F1:",
    round(
        optimized_validation_f1,
        4
    )
)

# ============================================================
# FINAL UNTOUCHED TEST SET
# ============================================================

print("\n" + "=" * 80)
print("FINAL OPTUNA-LoRA TEST EVALUATION")
print("=" * 80)


test_labels, \
test_probabilities, \
_ = evaluate_model(
    model,
    test_loader
)


# Default threshold
predictions_05 = (
    test_probabilities
    >= 0.50
).astype(int)


f1_default = f1_score(
    test_labels,
    predictions_05,
    zero_division=0
)


# Validation-optimized threshold
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
        f1_default,
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

print("\n" + "=" * 80)
print("OPTUNA-LoRA INFERENCE LATENCY")
print("=" * 80)


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
# SAVE TEST PREDICTIONS
# ============================================================

pd.DataFrame(
    {
        "true_label":
            test_labels,

        "failure_probability":
            test_probabilities,

        "prediction":
            predictions
    }
).to_csv(
    OUTPUT_DIR
    / "05_optuna_lora_test_predictions.csv",
    index=False
)

# ============================================================
# FINAL RESULTS
# ============================================================

results = {

    "model":
        "Optuna_LoRA_DistilBERT",

    "best_trial":
        int(
            study.best_trial.number
        ),

    "optuna_trials":
        N_TRIALS,

    "optuna_search_time_sec":
        float(
            optuna_search_time
        ),

    "learning_rate":
        float(
            best_params[
                "learning_rate"
            ]
        ),

    "weight_decay":
        float(
            best_params[
                "weight_decay"
            ]
        ),

    "lora_rank":
        int(
            best_params[
                "lora_rank"
            ]
        ),

    "lora_alpha":
        int(
            best_params[
                "lora_alpha"
            ]
        ),

    "lora_dropout":
        float(
            best_params[
                "lora_dropout"
            ]
        ),

    "best_epoch":
        int(
            best_epoch
        ),

    "total_parameters":
        int(
            total_parameters
        ),

    "trainable_parameters":
        int(
            trainable_parameters_count
        ),

    "trainable_percentage":
        float(
            trainable_percentage
        ),

    "adapter_size_mb":
        float(
            adapter_size_mb
        ),

    "final_training_time_sec":
        float(
            final_training_time
        ),

    "validation_f1_at_0.50":
        float(
            best_validation_f1_05
        ),

    "optimized_validation_f1":
        float(
            optimized_validation_f1
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

    "test_f1_at_0.50":
        float(
            f1_default
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
        )
}


with open(
    OUTPUT_DIR
    / "05_optuna_lora_final_results.json",
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
    / "05_optuna_lora_final_results.csv",
    index=False
)

# ============================================================
# CREATE ABLATION / COMPARISON TABLE
# ============================================================

comparison_rows = []

# ------------------------------------------------------------
# Baseline DistilBERT
# ------------------------------------------------------------

distilbert_file = (
    BASE_DIR
    / "results_distilbert"
    / "03_distilbert_final_results.json"
)

if distilbert_file.exists():

    with open(
        distilbert_file,
        "r"
    ) as file:

        d = json.load(
            file
        )

    comparison_rows.append(
        {
            "Model":
                "DistilBERT",

            "Trainable_Params":
                d[
                    "trainable_parameters"
                ],

            "Trainable_Percent":
                d[
                    "trainable_percentage"
                ],

            "Training_Time_Sec":
                d[
                    "training_time_sec"
                ],

            "F1":
                d[
                    "test_f1"
                ],

            "ROC_AUC":
                d[
                    "test_roc_auc"
                ],

            "PR_AUC":
                d[
                    "test_pr_auc"
                ],

            "Mean_Latency_ms":
                d[
                    "mean_latency_ms"
                ]
        }
    )

# ------------------------------------------------------------
# Fixed LoRA
# ------------------------------------------------------------

lora_file = (
    BASE_DIR
    / "results_lora_finetuned_base"
    / "04B_final_results.json"
)

if lora_file.exists():

    with open(
        lora_file,
        "r"
    ) as file:

        l = json.load(
            file
        )

    comparison_rows.append(
        {
            "Model":
                "LoRA-DistilBERT",

            "Trainable_Params":
                l[
                    "trainable_parameters"
                ],

            "Trainable_Percent":
                l[
                    "trainable_percentage"
                ],

            "Training_Time_Sec":
                l[
                    "training_time_sec"
                ],

            "F1":
                l[
                    "test_f1"
                ],

            "ROC_AUC":
                l[
                    "roc_auc"
                ],

            "PR_AUC":
                l[
                    "pr_auc"
                ],

            "Mean_Latency_ms":
                l[
                    "mean_latency_ms"
                ]
        }
    )

# ------------------------------------------------------------
# Optuna LoRA
# ------------------------------------------------------------

comparison_rows.append(
    {
        "Model":
            "Optuna-LoRA-DistilBERT",

        "Trainable_Params":
            trainable_parameters_count,

        "Trainable_Percent":
            trainable_percentage,

        "Training_Time_Sec":
            final_training_time,

        "F1":
            test_f1,

        "ROC_AUC":
            roc_auc,

        "PR_AUC":
            pr_auc,

        "Mean_Latency_ms":
            mean_latency
    }
)


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    OUTPUT_DIR
    / "05_distilbert_lora_optuna_comparison.csv",
    index=False
)


print("\n" + "=" * 80)
print("DISTILBERT / LoRA / OPTUNA-LoRA COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OPTUNA-LoRA FINAL SUMMARY")
print("=" * 80)

print(
    "Best trial:",
    study.best_trial.number
)

print(
    "Learning rate:",
    best_params[
        "learning_rate"
    ]
)

print(
    "Weight decay:",
    best_params[
        "weight_decay"
    ]
)

print(
    "LoRA rank:",
    best_params[
        "lora_rank"
    ]
)

print(
    "LoRA alpha:",
    best_params[
        "lora_alpha"
    ]
)

print(
    "LoRA dropout:",
    round(
        best_params[
            "lora_dropout"
        ],
        4
    )
)

print(
    "Trainable parameters:",
    f"{trainable_parameters_count:,}"
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
    "Best final epoch:",
    best_epoch
)

print(
    "Validation F1 @ 0.50:",
    round(
        best_validation_f1_05,
        4
    )
)

print(
    "Optimized validation F1:",
    round(
        optimized_validation_f1,
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
        test_f1,
        4
    )
)

print(
    "Test F1 @ 0.50:",
    round(
        f1_default,
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
    "Final training time:",
    round(
        final_training_time,
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
    "\nResults:",
    OUTPUT_DIR
)

print(
    "\nSTEP 8 COMPLETED SUCCESSFULLY"
)