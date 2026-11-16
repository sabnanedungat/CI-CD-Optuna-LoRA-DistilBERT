import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)

# ============================================================
# STEP 14
# PARTIAL-LOG OPTUNA-LoRA EVALUATION
#
# 25%, 50%, 75%, 100%
#
# IMPORTANT:
# - No model retraining.
# - Threshold selected only on validation data.
# - Test set remains evaluation-only.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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

BASE_MODEL_DIR = (
    BASE_DIR
    / "results_distilbert"
    / "best_distilbert_model"
)

LORA_MODEL_DIR = (
    BASE_DIR
    / "results_optuna_lora"
    / "best_optuna_lora_adapter"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results_partial_logs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_LENGTH = 128
BATCH_SIZE = 16

LOG_LEVELS = {
    "25%": "log_25",
    "50%": "log_50",
    "75%": "log_75",
    "100%": "log_100"
}

print("=" * 90)
print("STEP 14 - PARTIAL-LOG OPTUNA-LoRA EVALUATION")
print("=" * 90)

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
    VAL_FILE,
    TEST_FILE,
    BASE_MODEL_DIR,
    LORA_MODEL_DIR
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )

# ============================================================
# LOAD DATA
# ============================================================

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print(
    "\nValidation rows:",
    len(val_df)
)

print(
    "Test rows:",
    len(test_df)
)

# ============================================================
# VERIFY PARTIAL LOG COLUMNS
# ============================================================

for level, column in LOG_LEVELS.items():

    if column not in val_df.columns:

        raise ValueError(
            f"{column} missing from validation split."
        )

    if column not in test_df.columns:

        raise ValueError(
            f"{column} missing from test split."
        )

# ============================================================
# TOKENIZER + MODEL
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        LORA_MODEL_DIR
    )
)

print(
    "Loading DistilBERT base..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)

print(
    "Loading Optuna-LoRA adapter..."
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_DIR
)

model.to(
    device
)

model.eval()

# ============================================================
# DATASET CLASS
# ============================================================

class PartialLogDataset(Dataset):

    def __init__(
        self,
        dataframe,
        text_column
    ):

        self.texts = (
            dataframe[
                text_column
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe[
                "label"
            ]
            .astype(int)
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

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_probabilities(
    dataframe,
    text_column
):

    dataset = PartialLogDataset(
        dataframe,
        text_column
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    labels_all = []
    probabilities_all = []

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

            probabilities = (
                torch.softmax(
                    outputs.logits,
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
        )
    )

# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    rows = []

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.05,
        0.951,
        0.01
    ):

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        f1 = f1_score(
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

                "accuracy":
                    float(
                        accuracy_score(
                            labels,
                            predictions
                        )
                    ),

                "precision":
                    float(
                        precision_score(
                            labels,
                            predictions,
                            zero_division=0
                        )
                    ),

                "recall":
                    float(
                        recall_score(
                            labels,
                            predictions,
                            zero_division=0
                        )
                    ),

                "f1":
                    float(
                        f1
                    )
            }
        )

        if f1 > best_f1:

            best_f1 = (
                f1
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
# METRIC FUNCTION
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

    mcc = matthews_corrcoef(
        labels,
        predictions
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    pr_auc = average_precision_score(
        labels,
        probabilities
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
                roc_auc
            ),

        "pr_auc":
            float(
                pr_auc
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

# ============================================================
# RUN PARTIAL-LOG EXPERIMENT
# ============================================================

results = []

all_prediction_files = []

for level, text_column in LOG_LEVELS.items():

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"LOG AVAILABILITY: {level}"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_labels, \
    val_probabilities = predict_probabilities(
        val_df,
        text_column
    )

    best_threshold, \
    validation_f1, \
    threshold_df = find_best_threshold(
        val_labels,
        val_probabilities
    )

    threshold_df.to_csv(
        OUTPUT_DIR
        / f"09_threshold_search_{level.replace('%','pct')}.csv",
        index=False
    )

    print(
        "Validation-selected threshold:",
        round(
            best_threshold,
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

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_labels, \
    test_probabilities = predict_probabilities(
        test_df,
        text_column
    )

    metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        best_threshold
    )

    metrics[
        "log_availability"
    ] = level

    results.append(
        metrics
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

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

    prediction_file = (
        OUTPUT_DIR
        / f"09_predictions_{level.replace('%','pct')}.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    all_prediction_files.append(
        prediction_file
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nTest Accuracy:",
        round(
            metrics[
                "accuracy"
            ],
            4
        )
    )

    print(
        "Test Precision:",
        round(
            metrics[
                "precision"
            ],
            4
        )
    )

    print(
        "Test Recall:",
        round(
            metrics[
                "recall"
            ],
            4
        )
    )

    print(
        "Test F1:",
        round(
            metrics[
                "f1"
            ],
            4
        )
    )

    print(
        "Test ROC-AUC:",
        round(
            metrics[
                "roc_auc"
            ],
            4
        )
    )

    print(
        "Test PR-AUC:",
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
        "TN:",
        metrics[
            "tn"
        ]
    )

    print(
        "FP:",
        metrics[
            "fp"
        ]
    )

    print(
        "FN:",
        metrics[
            "fn"
        ]
    )

    print(
        "TP:",
        metrics[
            "tp"
        ]
    )

# ============================================================
# FINAL TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

column_order = [
    "log_availability",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "npv",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp"
]

results_df = results_df[
    column_order
]

results_df.to_csv(
    OUTPUT_DIR
    / "09_partial_log_full_results.csv",
    index=False
)

# ============================================================
# PAPER-FRIENDLY TABLE
# ============================================================

paper_df = results_df[
    [
        "log_availability",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]
].copy()

for column in [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]:

    paper_df[
        column
    ] = (
        paper_df[
            column
        ]
        .round(4)
    )

paper_df.to_csv(
    OUTPUT_DIR
    / "09_partial_log_paper_table.csv",
    index=False
)

# ============================================================
# EARLY-PREDICTION RETENTION
# ============================================================

full_f1 = float(
    results_df.loc[
        results_df[
            "log_availability"
        ] == "100%",
        "f1"
    ].iloc[0]
)

full_pr_auc = float(
    results_df.loc[
        results_df[
            "log_availability"
        ] == "100%",
        "pr_auc"
    ].iloc[0]
)

retention_rows = []

for _, row in results_df.iterrows():

    retention_rows.append(
        {
            "log_availability":
                row[
                    "log_availability"
                ],

            "f1":
                row[
                    "f1"
                ],

            "f1_retention_percent":
                (
                    row[
                        "f1"
                    ]
                    / full_f1
                    * 100
                    if full_f1 > 0
                    else np.nan
                ),

            "pr_auc":
                row[
                    "pr_auc"
                ],

            "pr_auc_retention_percent":
                (
                    row[
                        "pr_auc"
                    ]
                    / full_pr_auc
                    * 100
                    if full_pr_auc > 0
                    else np.nan
                )
        }
    )

retention_df = pd.DataFrame(
    retention_rows
)

retention_df.to_csv(
    OUTPUT_DIR
    / "09_partial_log_retention.csv",
    index=False
)

# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PARTIAL-LOG FINAL COMPARISON"
)

print(
    "=" * 90
)

print(
    paper_df.to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 90
)

print(
    "EARLY-PREDICTION PERFORMANCE RETENTION"
)

print(
    "=" * 90
)

retention_print = (
    retention_df.copy()
)

for column in [
    "f1",
    "f1_retention_percent",
    "pr_auc",
    "pr_auc_retention_percent"
]:

    retention_print[
        column
    ] = (
        retention_print[
            column
        ]
        .round(2)
    )

print(
    retention_print.to_string(
        index=False
    )
)

# ============================================================
# BEST EARLY POINT
# ============================================================

early_df = (
    results_df[
        results_df[
            "log_availability"
        ] != "100%"
    ]
    .copy()
)

best_early = (
    early_df.loc[
        early_df[
            "f1"
        ].idxmax()
    ]
)

print(
    "\n"
    + "=" * 90
)

print(
    "BEST EARLY-PREDICTION POINT"
)

print(
    "=" * 90
)

print(
    "Log availability:",
    best_early[
        "log_availability"
    ]
)

print(
    "F1:",
    round(
        best_early[
            "f1"
        ],
        4
    )
)

print(
    "ROC-AUC:",
    round(
        best_early[
            "roc_auc"
        ],
        4
    )
)

print(
    "PR-AUC:",
    round(
        best_early[
            "pr_auc"
        ],
        4
    )
)

print(
    "\nResults folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSTEP 14 COMPLETED SUCCESSFULLY"
)