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
    get_peft_model
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
# STEP 13B2
# PROJECT-DISJOINT OPTUNA-LoRA SEARCH
#
# Optuna sees ONLY:
#   11 inner-training projects
#   3 inner-validation projects
#
# Official validation = NEVER loaded
# Official test       = NEVER loaded
# ============================================================

SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16

N_TRIALS = 8
TRIAL_EPOCHS = 2

BASE_DIR = Path(__file__).resolve().parent

NESTED_DIR = (
    BASE_DIR
    / "results_cross_project"
    / "nested_optuna"
)

INNER_TRAIN_FILE = (
    NESTED_DIR
    / "inner_train.csv"
)

INNER_VAL_FILE = (
    NESTED_DIR
    / "inner_optuna_validation.csv"
)

INNER_BASE_MODEL = (
    NESTED_DIR
    / "inner_distilbert"
    / "best_inner_distilbert"
)

OUTPUT_DIR = (
    NESTED_DIR
    / "optuna_search"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 90)
print("STEP 13B2 - PROJECT-DISJOINT OPTUNA-LoRA SEARCH")
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
# CHECK INPUTS
# ============================================================

for path in [
    INNER_TRAIN_FILE,
    INNER_VAL_FILE,
    INNER_BASE_MODEL
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input:\n{path}"
        )

# ============================================================
# LOAD INNER DATA ONLY
# ============================================================

train_df = pd.read_csv(
    INNER_TRAIN_FILE,
    low_memory=False
)

val_df = pd.read_csv(
    INNER_VAL_FILE,
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
    "\nInner validation rows:",
    len(val_df)
)

print(
    "Inner validation projects:",
    val_df[
        "project_name"
    ].nunique()
)

# ============================================================
# VERIFY PROJECT DISJOINTNESS
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
        f"Project overlap detected: {overlap}"
    )

print(
    "\nProject overlap:",
    len(overlap)
)

print(
    "Project-disjoint check: PASSED"
)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        INNER_BASE_MODEL
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


train_dataset = CICDDataset(
    train_df
)

val_dataset = CICDDataset(
    val_df
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

    return (
        train_loader,
        val_loader
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

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

print(
    "\nClass weights:",
    class_weights
    .detach()
    .cpu()
    .numpy()
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
            INNER_BASE_MODEL
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
# VALIDATION
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
# OPTUNA OBJECTIVE
#
# IMPORTANT:
# Hyperparameter selection uses PR-AUC.
#
# No threshold optimization occurs here.
# Official validation and test are not accessed.
# ============================================================

def objective(
    trial
):

    learning_rate = trial.suggest_float(
        "learning_rate",
        2e-5,
        1.5e-4,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-5,
        0.02,
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
        0.15
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"TRIAL {trial.number}"
    )

    print(
        "=" * 90
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
    val_loader = create_loaders()

    model = create_lora_model(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout
    )

    trainable_parameters = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable_parameters
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

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

    best_pr_auc = -1.0

    best_f1 = None
    best_roc_auc = None
    best_epoch = None

    for epoch in range(
        1,
        TRIAL_EPOCHS + 1
    ):

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

        predictions_05 = (
            val_probabilities
            >= 0.50
        ).astype(int)

        val_f1 = f1_score(
            val_labels,
            predictions_05,
            zero_division=0
        )

        val_roc_auc = roc_auc_score(
            val_labels,
            val_probabilities
        )

        val_pr_auc = average_precision_score(
            val_labels,
            val_probabilities
        )

        print(
            f"\nEpoch {epoch}/{TRIAL_EPOCHS}"
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
            "F1 @ 0.50:",
            round(
                val_f1,
                4
            )
        )

        print(
            "ROC-AUC:",
            round(
                val_roc_auc,
                4
            )
        )

        print(
            "PR-AUC:",
            round(
                val_pr_auc,
                4
            )
        )

        if (
            val_pr_auc
            > best_pr_auc
        ):

            best_pr_auc = (
                val_pr_auc
            )

            best_f1 = (
                val_f1
            )

            best_roc_auc = (
                val_roc_auc
            )

            best_epoch = (
                epoch
            )

        # ----------------------------------------------------
        # Report PR-AUC to Optuna
        # ----------------------------------------------------

        trial.report(
            val_pr_auc,
            step=epoch
        )

        if trial.should_prune():

            print(
                "\nTrial pruned."
            )

            del model

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            raise optuna.TrialPruned()

    # ========================================================
    # SAVE EXTRA INFORMATION INSIDE TRIAL
    # ========================================================

    trial.set_user_attr(
        "best_epoch",
        int(
            best_epoch
        )
    )

    trial.set_user_attr(
        "best_f1_at_0.50",
        float(
            best_f1
        )
    )

    trial.set_user_attr(
        "best_roc_auc",
        float(
            best_roc_auc
        )
    )

    trial.set_user_attr(
        "trainable_parameters",
        int(
            trainable_count
        )
    )

    trial.set_user_attr(
        "total_parameters",
        int(
            total_count
        )
    )

    trial.set_user_attr(
        "trainable_percentage",
        float(
            trainable_count
            / total_count
            * 100
        )
    )

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return float(
        best_pr_auc
    )

# ============================================================
# OPTUNA STUDY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "STARTING PROJECT-DISJOINT OPTUNA SEARCH"
)

print(
    "=" * 90
)

print(
    "Trials:",
    N_TRIALS
)

print(
    "Epochs per trial:",
    TRIAL_EPOCHS
)

print(
    "Optimization objective: PR-AUC"
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
    study_name="nested_cross_project_lora"
)

search_start = (
    time.perf_counter()
)

study.optimize(
    objective,
    n_trials=N_TRIALS
)

search_time = (
    time.perf_counter()
    - search_start
)

# ============================================================
# SAVE COMPLETE TRIAL TABLE
# ============================================================

trials_df = (
    study
    .trials_dataframe()
)

trials_df.to_csv(
    OUTPUT_DIR
    / "08E_all_optuna_trials.csv",
    index=False
)

# ============================================================
# BEST TRIAL
# ============================================================

best_trial = (
    study.best_trial
)

best_params = (
    study.best_params
)

best_info = {

    "best_trial":
        int(
            best_trial.number
        ),

    "optimization_metric":
        "PR-AUC",

    "best_inner_validation_pr_auc":
        float(
            study.best_value
        ),

    "best_inner_validation_f1_at_0.50":
        float(
            best_trial.user_attrs[
                "best_f1_at_0.50"
            ]
        ),

    "best_inner_validation_roc_auc":
        float(
            best_trial.user_attrs[
                "best_roc_auc"
            ]
        ),

    "best_trial_epoch":
        int(
            best_trial.user_attrs[
                "best_epoch"
            ]
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

    "trainable_parameters":
        int(
            best_trial.user_attrs[
                "trainable_parameters"
            ]
        ),

    "total_parameters":
        int(
            best_trial.user_attrs[
                "total_parameters"
            ]
        ),

    "trainable_percentage":
        float(
            best_trial.user_attrs[
                "trainable_percentage"
            ]
        ),

    "n_trials":
        N_TRIALS,

    "trial_epochs":
        TRIAL_EPOCHS,

    "search_time_sec":
        float(
            search_time
        )
}

with open(
    OUTPUT_DIR
    / "08E_best_optuna_parameters.json",
    "w"
) as file:

    json.dump(
        best_info,
        file,
        indent=4
    )

pd.DataFrame(
    [
        best_info
    ]
).to_csv(
    OUTPUT_DIR
    / "08E_best_optuna_parameters.csv",
    index=False
)

# ============================================================
# FINAL DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PROJECT-DISJOINT OPTUNA BEST RESULT"
)

print(
    "=" * 90
)

print(
    "Best trial:",
    best_info[
        "best_trial"
    ]
)

print(
    "Best inner-validation PR-AUC:",
    round(
        best_info[
            "best_inner_validation_pr_auc"
        ],
        4
    )
)

print(
    "F1 @ 0.50:",
    round(
        best_info[
            "best_inner_validation_f1_at_0.50"
        ],
        4
    )
)

print(
    "ROC-AUC:",
    round(
        best_info[
            "best_inner_validation_roc_auc"
        ],
        4
    )
)

print(
    "Best trial epoch:",
    best_info[
        "best_trial_epoch"
    ]
)

print(
    "\nBest hyperparameters:"
)

print(
    "Learning rate:",
    best_info[
        "learning_rate"
    ]
)

print(
    "Weight decay:",
    best_info[
        "weight_decay"
    ]
)

print(
    "LoRA rank:",
    best_info[
        "lora_rank"
    ]
)

print(
    "LoRA alpha:",
    best_info[
        "lora_alpha"
    ]
)

print(
    "LoRA dropout:",
    round(
        best_info[
            "lora_dropout"
        ],
        4
    )
)

print(
    "\nTrainable parameters:",
    f"{best_info['trainable_parameters']:,}"
)

print(
    "Trainable percentage:",
    f"{best_info['trainable_percentage']:.4f}%"
)

print(
    "Search time:",
    round(
        search_time,
        2
    ),
    "seconds"
)

print(
    "\nOfficial validation projects used: NO"
)

print(
    "Official test projects used: NO"
)

print(
    "\nResults:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSTEP 13B2 COMPLETED SUCCESSFULLY"
)