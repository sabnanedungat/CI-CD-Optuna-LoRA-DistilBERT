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
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef
)

# ============================================================
# STEP 13C
# FINAL CROSS-PROJECT OPTUNA-LoRA
# ============================================================

SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16

MAX_EPOCHS = 3
PATIENCE = 2

BASE_DIR = Path(__file__).resolve().parent

CROSS_DIR = (
    BASE_DIR
    / "results_cross_project"
)

TRAIN_FILE = (
    CROSS_DIR
    / "cross_project_train.csv"
)

VAL_FILE = (
    CROSS_DIR
    / "cross_project_validation.csv"
)

TEST_FILE = (
    CROSS_DIR
    / "cross_project_test.csv"
)

BASE_MODEL_DIR = (
    CROSS_DIR
    / "distilbert"
    / "best_cross_project_distilbert"
)

OPTUNA_PARAM_FILE = (
    CROSS_DIR
    / "nested_optuna"
    / "optuna_search"
    / "08E_best_optuna_parameters.json"
)

OUTPUT_DIR = (
    CROSS_DIR
    / "final_optuna_lora"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "best_final_optuna_lora"
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
print("STEP 13C - FINAL CROSS-PROJECT OPTUNA-LoRA")
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
# CHECK REQUIRED FILES
# ============================================================

for path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BASE_MODEL_DIR,
    OPTUNA_PARAM_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )

# ============================================================
# LOAD OPTUNA PARAMETERS
# ============================================================

with open(
    OPTUNA_PARAM_FILE,
    "r"
) as file:

    best_params = json.load(
        file
    )

LEARNING_RATE = float(
    best_params[
        "learning_rate"
    ]
)

WEIGHT_DECAY = float(
    best_params[
        "weight_decay"
    ]
)

LORA_RANK = int(
    best_params[
        "lora_rank"
    ]
)

LORA_ALPHA = int(
    best_params[
        "lora_alpha"
    ]
)

LORA_DROPOUT = float(
    best_params[
        "lora_dropout"
    ]
)

print(
    "\nBest nested Optuna parameters:"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "LoRA rank:",
    LORA_RANK
)

print(
    "LoRA alpha:",
    LORA_ALPHA
)

print(
    "LoRA dropout:",
    LORA_DROPOUT
)

# ============================================================
# LOAD ONLY TRAIN + OFFICIAL VALIDATION FIRST
#
# TEST SET IS DELIBERATELY NOT LOADED YET.
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
    "\nDevelopment training rows:",
    len(train_df)
)

print(
    "Development training projects:",
    train_df[
        "project_name"
    ].nunique()
)

print(
    "\nOfficial validation rows:",
    len(val_df)
)

print(
    "Official validation projects:",
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

assert len(
    train_projects
    & val_projects
) == 0

print(
    "\nTrain/validation project overlap: 0"
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

# ============================================================
# LOAD CROSS-PROJECT DISTILBERT BASE
# ============================================================

print(
    "\nLoading cross-project DistilBERT base..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)

# ============================================================
# APPLY OPTUNA LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(

    task_type=TaskType.SEQ_CLS,

    inference_mode=False,

    r=LORA_RANK,

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

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

trainable_percentage = (
    trainable_parameters
    / total_parameters
    * 100
)

print(
    "\n"
    + "=" * 90
)

print(
    "PARAMETER EFFICIENCY"
)

print(
    "=" * 90
)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
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
# TRAINABLE PARAMETERS
# ============================================================

trainable_parameter_list = [

    p

    for p in model.parameters()

    if p.requires_grad
]

optimizer = AdamW(

    trainable_parameter_list,

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
# TRAIN FINAL LoRA
#
# Best epoch chosen by official-validation PR-AUC.
# No test information is used.
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL LoRA TRAINING ON 14 DEVELOPMENT PROJECTS"
)

print(
    "=" * 90
)

best_validation_pr_auc = -1.0
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
            trainable_parameter_list,
            1.0
        )

        optimizer.step()

        scheduler.step()

        training_loss += (
            loss.item()
        )

    training_loss /= (
        len(train_loader)
    )

    # --------------------------------------------------------
    # OFFICIAL VALIDATION ONLY
    # --------------------------------------------------------

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

    val_accuracy_05 = accuracy_score(
        val_labels,
        val_predictions_05
    )

    val_roc_auc = roc_auc_score(
        val_labels,
        val_probabilities
    )

    val_pr_auc = average_precision_score(
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
            training_loss,
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
        "Validation accuracy @ 0.50:",
        round(
            val_accuracy_05,
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
        "Validation ROC-AUC:",
        round(
            val_roc_auc,
            4
        )
    )

    print(
        "Validation PR-AUC:",
        round(
            val_pr_auc,
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
                training_loss,

            "validation_loss":
                val_loss,

            "validation_accuracy_at_0.50":
                val_accuracy_05,

            "validation_f1_at_0.50":
                val_f1_05,

            "validation_roc_auc":
                val_roc_auc,

            "validation_pr_auc":
                val_pr_auc,

            "epoch_time_sec":
                epoch_time
        }
    )

    # --------------------------------------------------------
    # SELECT EPOCH USING PR-AUC
    # --------------------------------------------------------

    if (
        val_pr_auc
        > best_validation_pr_auc
    ):

        best_validation_pr_auc = (
            val_pr_auc
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
            "New best final LoRA model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "No PR-AUC improvement."
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
    / "08F_training_history.csv",
    index=False
)

# ============================================================
# RELOAD BEST FINAL MODEL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "RELOADING BEST FINAL OPTUNA-LoRA"
)

print(
    "=" * 90
)

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
# OFFICIAL VALIDATION THRESHOLD SEARCH
# ============================================================

val_labels, \
val_probabilities, \
_ = evaluate_model(
    model,
    val_loader
)

threshold_rows = []

for threshold in np.arange(
    0.05,
    0.951,
    0.01
):

    predictions = (
        val_probabilities
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            val_labels,
            predictions
        )
    )

    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        predictions,
        zero_division=0
    )

    threshold_rows.append(
        {
            "threshold":
                float(
                    threshold
                ),

            "accuracy":
                float(
                    accuracy
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                )
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    OUTPUT_DIR
    / "08F_validation_threshold_search.csv",
    index=False
)

# ============================================================
# THREE VALIDATION-SELECTED THRESHOLDS
# ============================================================

f1_best_row = (
    threshold_df
    .loc[
        threshold_df[
            "f1"
        ].idxmax()
    ]
)

accuracy_best_row = (
    threshold_df
    .loc[
        threshold_df[
            "accuracy"
        ].idxmax()
    ]
)

balanced_best_row = (
    threshold_df
    .loc[
        threshold_df[
            "balanced_accuracy"
        ].idxmax()
    ]
)

F1_THRESHOLD = float(
    f1_best_row[
        "threshold"
    ]
)

ACCURACY_THRESHOLD = float(
    accuracy_best_row[
        "threshold"
    ]
)

BALANCED_THRESHOLD = float(
    balanced_best_row[
        "threshold"
    ]
)

print(
    "\n"
    + "=" * 90
)

print(
    "OFFICIAL VALIDATION THRESHOLD SELECTION"
)

print(
    "=" * 90
)

print(
    "F1-optimal threshold:",
    round(
        F1_THRESHOLD,
        2
    )
)

print(
    "Validation F1:",
    round(
        f1_best_row[
            "f1"
        ],
        4
    )
)

print(
    "\nAccuracy-optimal threshold:",
    round(
        ACCURACY_THRESHOLD,
        2
    )
)

print(
    "Validation accuracy:",
    round(
        accuracy_best_row[
            "accuracy"
        ],
        4
    )
)

print(
    "\nBalanced-accuracy threshold:",
    round(
        BALANCED_THRESHOLD,
        2
    )
)

print(
    "Validation balanced accuracy:",
    round(
        balanced_best_row[
            "balanced_accuracy"
        ],
        4
    )
)

# ============================================================
# LOCK THRESHOLDS BEFORE TEST SET IS LOADED
# ============================================================

print(
    "\nThresholds locked."
)

print(
    "The test set has not been used for model or threshold selection."
)

# ============================================================
# NOW LOAD FINAL UNTOUCHED TEST PROJECTS
# ============================================================

print(
    "\nLoading untouched test projects..."
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

test_df[TEXT_COLUMN] = (
    test_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
)

test_df["label"] = (
    test_df["label"]
    .astype(int)
)

test_projects = set(
    test_df[
        "project_name"
    ].unique()
)

assert len(
    train_projects
    & test_projects
) == 0

assert len(
    val_projects
    & test_projects
) == 0

print(
    "Test rows:",
    len(test_df)
)

print(
    "Test projects:",
    len(test_projects)
)

print(
    "Train/test overlap: 0"
)

print(
    "Validation/test overlap: 0"
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
# TEST PROBABILITIES
# ============================================================

test_labels, \
test_probabilities, \
_ = evaluate_model(
    model,
    test_loader
)

test_roc_auc = roc_auc_score(
    test_labels,
    test_probabilities
)

test_pr_auc = average_precision_score(
    test_labels,
    test_probabilities
)

# ============================================================
# FULL METRIC FUNCTION
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions
        )
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    specificity = (
        tn
        / (
            tn + fp
        )
        if (
            tn + fp
        ) > 0
        else 0
    )

    npv = (
        tn
        / (
            tn + fn
        )
        if (
            tn + fn
        ) > 0
        else 0
    )

    fpr = (
        fp
        / (
            fp + tn
        )
        if (
            fp + tn
        ) > 0
        else 0
    )

    fnr = (
        fn
        / (
            fn + tp
        )
        if (
            fn + tp
        ) > 0
        else 0
    )

    mcc = matthews_corrcoef(
        labels,
        predictions
    )

    return {

        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "specificity":
            float(
                specificity
            ),

        "npv":
            float(
                npv
            ),

        "f1":
            float(
                f1
            ),

        "mcc":
            float(
                mcc
            ),

        "roc_auc":
            float(
                test_roc_auc
            ),

        "pr_auc":
            float(
                test_pr_auc
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),

        "false_positive_rate":
            float(
                fpr
            ),

        "false_negative_rate":
            float(
                fnr
            )
    }

# ============================================================
# TEST ALL PRE-LOCKED OPERATING POINTS
# ============================================================

default_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    0.50
)

f1_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    F1_THRESHOLD
)

accuracy_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    ACCURACY_THRESHOLD
)

balanced_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    BALANCED_THRESHOLD
)

# ============================================================
# DISPLAY FUNCTION
# ============================================================

def display_metrics(
    title,
    metrics
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )

    print(
        "Threshold:",
        round(
            metrics[
                "threshold"
            ],
            2
        )
    )

    print(
        "Accuracy:",
        round(
            metrics[
                "accuracy"
            ],
            4
        )
    )

    print(
        "Balanced accuracy:",
        round(
            metrics[
                "balanced_accuracy"
            ],
            4
        )
    )

    print(
        "Precision:",
        round(
            metrics[
                "precision"
            ],
            4
        )
    )

    print(
        "Recall / Sensitivity:",
        round(
            metrics[
                "recall"
            ],
            4
        )
    )

    print(
        "Specificity:",
        round(
            metrics[
                "specificity"
            ],
            4
        )
    )

    print(
        "NPV:",
        round(
            metrics[
                "npv"
            ],
            4
        )
    )

    print(
        "F1:",
        round(
            metrics[
                "f1"
            ],
            4
        )
    )

    print(
        "MCC:",
        round(
            metrics[
                "mcc"
            ],
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            metrics[
                "roc_auc"
            ],
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            metrics[
                "pr_auc"
            ],
            4
        )
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        "TN =",
        metrics[
            "tn"
        ],
        "| FP =",
        metrics[
            "fp"
        ]
    )

    print(
        "FN =",
        metrics[
            "fn"
        ],
        "| TP =",
        metrics[
            "tp"
        ]
    )

    print(
        "False positive rate:",
        round(
            metrics[
                "false_positive_rate"
            ],
            4
        )
    )

    print(
        "False negative rate:",
        round(
            metrics[
                "false_negative_rate"
            ],
            4
        )
    )

# ============================================================
# DISPLAY MAIN RESULTS
# ============================================================

display_metrics(
    "TEST - DEFAULT THRESHOLD 0.50",
    default_metrics
)

display_metrics(
    "TEST - VALIDATION F1-OPTIMAL THRESHOLD",
    f1_metrics
)

display_metrics(
    "TEST - VALIDATION ACCURACY-OPTIMAL THRESHOLD",
    accuracy_metrics
)

display_metrics(
    "TEST - VALIDATION BALANCED-ACCURACY THRESHOLD",
    balanced_metrics
)

# ============================================================
# PROJECT-WISE PERFORMANCE USING MAIN F1 THRESHOLD
# ============================================================

f1_predictions = (
    test_probabilities
    >= F1_THRESHOLD
).astype(int)

test_output = test_df[
    [
        "project_name",
        "label"
    ]
].copy()

test_output[
    "failure_probability"
] = test_probabilities

test_output[
    "prediction"
] = f1_predictions

project_rows = []

for project, group in test_output.groupby(
    "project_name"
):

    y = (
        group[
            "label"
        ]
        .to_numpy()
    )

    p = (
        group[
            "prediction"
        ]
        .to_numpy()
    )

    probabilities = (
        group[
            "failure_probability"
        ]
        .to_numpy()
    )

    cm = confusion_matrix(
        y,
        p
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        tn
        / (
            tn + fp
        )
        if (
            tn + fp
        ) > 0
        else 0
    )

    row = {

        "project":
            project,

        "n_builds":
            len(group),

        "failure_rate":
            float(
                y.mean()
            ),

        "accuracy":
            accuracy_score(
                y,
                p
            ),

        "precision":
            precision_score(
                y,
                p,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                p,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y,
                p,
                zero_division=0
            ),

        "mcc":
            matthews_corrcoef(
                y,
                p
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            )
    }

    if len(
        np.unique(
            y
        )
    ) == 2:

        row[
            "roc_auc"
        ] = roc_auc_score(
            y,
            probabilities
        )

        row[
            "pr_auc"
        ] = average_precision_score(
            y,
            probabilities
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
    + "=" * 90
)

print(
    "FINAL UNSEEN-PROJECT PERFORMANCE"
)

print(
    "=" * 90
)

print(
    project_df
    .round(4)
    .to_string(
        index=False
    )
)

# ============================================================
# ADAPTER SIZE
# ============================================================

adapter_bytes = sum(

    file.stat().st_size

    for file in MODEL_DIR.rglob(
        "*"
    )

    if file.is_file()
)

adapter_size_mb = (
    adapter_bytes
    / (
        1024 ** 2
    )
)

# ============================================================
# INFERENCE LATENCY
# ============================================================

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

p95_latency = float(
    np.percentile(
        latencies,
        95
    )
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_output.to_csv(
    OUTPUT_DIR
    / "08F_final_test_predictions.csv",
    index=False
)

project_df.to_csv(
    OUTPUT_DIR
    / "08F_project_level_metrics.csv",
    index=False
)

# ============================================================
# SAVE OPERATING POINT COMPARISON
# ============================================================

operating_points = pd.DataFrame(
    [
        {
            "Operating_Point":
                "Default 0.50",
            **default_metrics
        },

        {
            "Operating_Point":
                "Validation F1 Optimal",
            **f1_metrics
        },

        {
            "Operating_Point":
                "Validation Accuracy Optimal",
            **accuracy_metrics
        },

        {
            "Operating_Point":
                "Validation Balanced Accuracy Optimal",
            **balanced_metrics
        }
    ]
)

operating_points.to_csv(
    OUTPUT_DIR
    / "08F_operating_point_comparison.csv",
    index=False
)

# ============================================================
# COMPARE AGAINST CROSS-PROJECT DISTILBERT
# ============================================================

baseline_file = (
    CROSS_DIR
    / "distilbert"
    / "08B_cross_project_distilbert_results.json"
)

comparison_rows = []

if baseline_file.exists():

    with open(
        baseline_file,
        "r"
    ) as file:

        baseline = json.load(
            file
        )

    comparison_rows.append(
        {
            "Model":
                "Cross-project DistilBERT",

            "Accuracy":
                baseline[
                    "test_accuracy"
                ],

            "Precision":
                baseline[
                    "test_precision"
                ],

            "Recall":
                baseline[
                    "test_recall"
                ],

            "F1":
                baseline[
                    "test_f1"
                ],

            "ROC_AUC":
                baseline[
                    "test_roc_auc"
                ],

            "PR_AUC":
                baseline[
                    "test_pr_auc"
                ],

            "Trainable_Parameters":
                baseline[
                    "trainable_parameters"
                ]
        }
    )

comparison_rows.append(
    {
        "Model":
            "Cross-project Optuna-LoRA",

        "Accuracy":
            f1_metrics[
                "accuracy"
            ],

        "Precision":
            f1_metrics[
                "precision"
            ],

        "Recall":
            f1_metrics[
                "recall"
            ],

        "F1":
            f1_metrics[
                "f1"
            ],

        "ROC_AUC":
            f1_metrics[
                "roc_auc"
            ],

        "PR_AUC":
            f1_metrics[
                "pr_auc"
            ],

        "Trainable_Parameters":
            trainable_parameters
    }
)

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "08F_distilbert_vs_optuna_lora.csv",
    index=False
)

print(
    "\n"
    + "=" * 90
)

print(
    "CROSS-PROJECT MODEL COMPARISON"
)

print(
    "=" * 90
)

print(
    comparison_df
    .round(4)
    .to_string(
        index=False
    )
)

# ============================================================
# SAVE FINAL JSON
# ============================================================

final_results = {

    "model":
        "Final_Cross_Project_Optuna_LoRA",

    "training_projects":
        int(
            train_df[
                "project_name"
            ].nunique()
        ),

    "validation_projects":
        int(
            val_df[
                "project_name"
            ].nunique()
        ),

    "test_projects":
        int(
            test_df[
                "project_name"
            ].nunique()
        ),

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "lora_rank":
        LORA_RANK,

    "lora_alpha":
        LORA_ALPHA,

    "lora_dropout":
        LORA_DROPOUT,

    "best_epoch":
        int(
            best_epoch
        ),

    "validation_pr_auc":
        float(
            best_validation_pr_auc
        ),

    "f1_threshold":
        F1_THRESHOLD,

    "accuracy_threshold":
        ACCURACY_THRESHOLD,

    "balanced_accuracy_threshold":
        BALANCED_THRESHOLD,

    "total_parameters":
        int(
            total_parameters
        ),

    "trainable_parameters":
        int(
            trainable_parameters
        ),

    "trainable_percentage":
        float(
            trainable_percentage
        ),

    "adapter_size_mb":
        float(
            adapter_size_mb
        ),

    "training_time_sec":
        float(
            training_time
        ),

    "mean_latency_ms":
        mean_latency,

    "p95_latency_ms":
        p95_latency,

    "main_f1_operating_point":
        f1_metrics,

    "accuracy_operating_point":
        accuracy_metrics,

    "balanced_accuracy_operating_point":
        balanced_metrics
}

with open(
    OUTPUT_DIR
    / "08F_final_results.json",
    "w"
) as file:

    json.dump(
        final_results,
        file,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL CROSS-PROJECT OPTUNA-LoRA SUMMARY"
)

print(
    "=" * 90
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "LoRA rank:",
    LORA_RANK
)

print(
    "LoRA alpha:",
    LORA_ALPHA
)

print(
    "LoRA dropout:",
    round(
        LORA_DROPOUT,
        4
    )
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
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
    "\nMAIN F1-OPTIMIZED TEST RESULT"
)

print(
    "Threshold:",
    round(
        F1_THRESHOLD,
        2
    )
)

print(
    "Accuracy:",
    round(
        f1_metrics[
            "accuracy"
        ],
        4
    )
)

print(
    "Precision:",
    round(
        f1_metrics[
            "precision"
        ],
        4
    )
)

print(
    "Recall:",
    round(
        f1_metrics[
            "recall"
        ],
        4
    )
)

print(
    "Specificity:",
    round(
        f1_metrics[
            "specificity"
        ],
        4
    )
)

print(
    "F1:",
    round(
        f1_metrics[
            "f1"
        ],
        4
    )
)

print(
    "MCC:",
    round(
        f1_metrics[
            "mcc"
        ],
        4
    )
)

print(
    "ROC-AUC:",
    round(
        test_roc_auc,
        4
    )
)

print(
    "PR-AUC:",
    round(
        test_pr_auc,
        4
    )
)

print(
    "\nConfusion matrix:"
)

print(
    "TN:",
    f1_metrics[
        "tn"
    ]
)

print(
    "FP:",
    f1_metrics[
        "fp"
    ]
)

print(
    "FN:",
    f1_metrics[
        "fn"
    ]
)

print(
    "TP:",
    f1_metrics[
        "tp"
    ]
)

print(
    "\nACCURACY-OPTIMIZED TEST RESULT"
)

print(
    "Threshold:",
    round(
        ACCURACY_THRESHOLD,
        2
    )
)

print(
    "Accuracy:",
    round(
        accuracy_metrics[
            "accuracy"
        ],
        4
    )
)

print(
    "Recall:",
    round(
        accuracy_metrics[
            "recall"
        ],
        4
    )
)

print(
    "F1:",
    round(
        accuracy_metrics[
            "f1"
        ],
        4
    )
)

print(
    "\nTraining time:",
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
    "\nSTEP 13C COMPLETED SUCCESSFULLY"
)