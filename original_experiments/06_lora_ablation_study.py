import json
from pathlib import Path

import pandas as pd

# ============================================================
# STEP 9
# FORMAL ABLATION STUDY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DISTILBERT_FILE = (
    BASE_DIR
    / "results_distilbert"
    / "03_distilbert_final_results.json"
)

LORA_FILE = (
    BASE_DIR
    / "results_lora_finetuned_base"
    / "04B_final_results.json"
)

OPTUNA_FILE = (
    BASE_DIR
    / "results_optuna_lora"
    / "05_optuna_lora_final_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results_ablation"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

print("=" * 85)
print("STEP 9 - DISTILBERT / LoRA / OPTUNA / THRESHOLD ABLATION")
print("=" * 85)

# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    DISTILBERT_FILE,
    LORA_FILE,
    OPTUNA_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required result file not found:\n"
            f"{file_path}"
        )

# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    DISTILBERT_FILE,
    "r"
) as file:

    distilbert = json.load(
        file
    )


with open(
    LORA_FILE,
    "r"
) as file:

    lora = json.load(
        file
    )


with open(
    OPTUNA_FILE,
    "r"
) as file:

    optuna = json.load(
        file
    )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_percent_reduction(
    original,
    new
):

    if original == 0:
        return None

    return (
        (
            original - new
        )
        / original
        * 100
    )


# ============================================================
# BASELINE PARAMETER INFORMATION
# ============================================================

distilbert_trainable = (
    distilbert[
        "trainable_parameters"
    ]
)

lora_trainable = (
    lora[
        "trainable_parameters"
    ]
)

optuna_trainable = (
    optuna[
        "trainable_parameters"
    ]
)


lora_parameter_reduction = (
    safe_percent_reduction(
        distilbert_trainable,
        lora_trainable
    )
)


optuna_parameter_reduction = (
    safe_percent_reduction(
        distilbert_trainable,
        optuna_trainable
    )
)

# ============================================================
# TRAINING TIME REDUCTION
# ============================================================

distilbert_training_time = (
    distilbert[
        "training_time_sec"
    ]
)

lora_training_time = (
    lora[
        "training_time_sec"
    ]
)

optuna_final_training_time = (
    optuna[
        "final_training_time_sec"
    ]
)


lora_training_reduction = (
    safe_percent_reduction(
        distilbert_training_time,
        lora_training_time
    )
)


optuna_training_reduction = (
    safe_percent_reduction(
        distilbert_training_time,
        optuna_final_training_time
    )
)

# ============================================================
# CREATE ABLATION TABLE
# ============================================================

rows = []

# ------------------------------------------------------------
# A. Full DistilBERT - default threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "A",

        "Model":
            "DistilBERT",

        "LoRA":
            "No",

        "Optuna":
            "No",

        "Threshold_Optimization":
            "No",

        "Threshold":
            0.50,

        "Test_F1":
            distilbert[
                "test_f1_default_threshold"
            ],

        "ROC_AUC":
            distilbert[
                "test_roc_auc"
            ],

        "PR_AUC":
            distilbert[
                "test_pr_auc"
            ],

        "Trainable_Parameters":
            distilbert_trainable,

        "Trainable_Percent":
            distilbert[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            distilbert_training_time,

        "Mean_Latency_ms":
            distilbert[
                "mean_latency_ms"
            ]
    }
)

# ------------------------------------------------------------
# B. DistilBERT + validation threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "B",

        "Model":
            "DistilBERT",

        "LoRA":
            "No",

        "Optuna":
            "No",

        "Threshold_Optimization":
            "Yes",

        "Threshold":
            distilbert[
                "optimized_threshold"
            ],

        "Test_F1":
            distilbert[
                "test_f1"
            ],

        "ROC_AUC":
            distilbert[
                "test_roc_auc"
            ],

        "PR_AUC":
            distilbert[
                "test_pr_auc"
            ],

        "Trainable_Parameters":
            distilbert_trainable,

        "Trainable_Percent":
            distilbert[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            distilbert_training_time,

        "Mean_Latency_ms":
            distilbert[
                "mean_latency_ms"
            ]
    }
)

# ------------------------------------------------------------
# C. Fixed LoRA - default threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "C",

        "Model":
            "LoRA-DistilBERT",

        "LoRA":
            "Yes",

        "Optuna":
            "No",

        "Threshold_Optimization":
            "No",

        "Threshold":
            0.50,

        "Test_F1":
            lora[
                "test_f1_default_threshold"
            ],

        "ROC_AUC":
            lora[
                "roc_auc"
            ],

        "PR_AUC":
            lora[
                "pr_auc"
            ],

        "Trainable_Parameters":
            lora_trainable,

        "Trainable_Percent":
            lora[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            lora_training_time,

        "Mean_Latency_ms":
            lora[
                "mean_latency_ms"
            ]
    }
)

# ------------------------------------------------------------
# D. Fixed LoRA + threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "D",

        "Model":
            "LoRA-DistilBERT",

        "LoRA":
            "Yes",

        "Optuna":
            "No",

        "Threshold_Optimization":
            "Yes",

        "Threshold":
            lora[
                "threshold"
            ],

        "Test_F1":
            lora[
                "test_f1"
            ],

        "ROC_AUC":
            lora[
                "roc_auc"
            ],

        "PR_AUC":
            lora[
                "pr_auc"
            ],

        "Trainable_Parameters":
            lora_trainable,

        "Trainable_Percent":
            lora[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            lora_training_time,

        "Mean_Latency_ms":
            lora[
                "mean_latency_ms"
            ]
    }
)

# ------------------------------------------------------------
# E. Optuna-LoRA - default threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "E",

        "Model":
            "Optuna-LoRA-DistilBERT",

        "LoRA":
            "Yes",

        "Optuna":
            "Yes",

        "Threshold_Optimization":
            "No",

        "Threshold":
            0.50,

        "Test_F1":
            optuna[
                "test_f1_at_0.50"
            ],

        "ROC_AUC":
            optuna[
                "test_roc_auc"
            ],

        "PR_AUC":
            optuna[
                "test_pr_auc"
            ],

        "Trainable_Parameters":
            optuna_trainable,

        "Trainable_Percent":
            optuna[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            optuna_final_training_time,

        "Mean_Latency_ms":
            optuna[
                "mean_latency_ms"
            ]
    }
)

# ------------------------------------------------------------
# F. Optuna-LoRA + threshold
# ------------------------------------------------------------

rows.append(
    {
        "Experiment":
            "F",

        "Model":
            "Optuna-LoRA-DistilBERT",

        "LoRA":
            "Yes",

        "Optuna":
            "Yes",

        "Threshold_Optimization":
            "Yes",

        "Threshold":
            optuna[
                "optimized_threshold"
            ],

        "Test_F1":
            optuna[
                "test_f1"
            ],

        "ROC_AUC":
            optuna[
                "test_roc_auc"
            ],

        "PR_AUC":
            optuna[
                "test_pr_auc"
            ],

        "Trainable_Parameters":
            optuna_trainable,

        "Trainable_Percent":
            optuna[
                "trainable_percentage"
            ],

        "Training_Time_Sec":
            optuna_final_training_time,

        "Mean_Latency_ms":
            optuna[
                "mean_latency_ms"
            ]
    }
)

# ============================================================
# DATAFRAME
# ============================================================

ablation_df = pd.DataFrame(
    rows
)

# ============================================================
# CALCULATE F1 CHANGES
# ============================================================

distilbert_default_f1 = (
    distilbert[
        "test_f1_default_threshold"
    ]
)

distilbert_threshold_f1 = (
    distilbert[
        "test_f1"
    ]
)

lora_default_f1 = (
    lora[
        "test_f1_default_threshold"
    ]
)

lora_threshold_f1 = (
    lora[
        "test_f1"
    ]
)

optuna_default_f1 = (
    optuna[
        "test_f1_at_0.50"
    ]
)

optuna_threshold_f1 = (
    optuna[
        "test_f1"
    ]
)

# ============================================================
# COMPONENT EFFECT TABLE
# ============================================================

effects = [

    {
        "Comparison":
            "DistilBERT threshold optimization",

        "Before_F1":
            distilbert_default_f1,

        "After_F1":
            distilbert_threshold_f1,

        "Delta_F1":
            (
                distilbert_threshold_f1
                - distilbert_default_f1
            )
    },

    {
        "Comparison":
            "DistilBERT -> Fixed LoRA at default threshold",

        "Before_F1":
            distilbert_default_f1,

        "After_F1":
            lora_default_f1,

        "Delta_F1":
            (
                lora_default_f1
                - distilbert_default_f1
            )
    },

    {
        "Comparison":
            "Fixed LoRA threshold optimization",

        "Before_F1":
            lora_default_f1,

        "After_F1":
            lora_threshold_f1,

        "Delta_F1":
            (
                lora_threshold_f1
                - lora_default_f1
            )
    },

    {
        "Comparison":
            "Fixed LoRA -> Optuna-LoRA at default threshold",

        "Before_F1":
            lora_default_f1,

        "After_F1":
            optuna_default_f1,

        "Delta_F1":
            (
                optuna_default_f1
                - lora_default_f1
            )
    },

    {
        "Comparison":
            "Optuna-LoRA threshold optimization",

        "Before_F1":
            optuna_default_f1,

        "After_F1":
            optuna_threshold_f1,

        "Delta_F1":
            (
                optuna_threshold_f1
                - optuna_default_f1
            )
    },

    {
        "Comparison":
            "DistilBERT default -> Final Optuna-LoRA",

        "Before_F1":
            distilbert_default_f1,

        "After_F1":
            optuna_threshold_f1,

        "Delta_F1":
            (
                optuna_threshold_f1
                - distilbert_default_f1
            )
    }
]


effects_df = pd.DataFrame(
    effects
)

# ============================================================
# EFFICIENCY SUMMARY
# ============================================================

efficiency_df = pd.DataFrame(
    [
        {
            "Model":
                "LoRA-DistilBERT",

            "Trainable_Parameter_Reduction_Percent":
                lora_parameter_reduction,

            "Training_Time_Reduction_Percent":
                lora_training_reduction,

            "Adapter_Size_MB":
                lora[
                    "adapter_size_mb"
                ]
        },

        {
            "Model":
                "Optuna-LoRA-DistilBERT",

            "Trainable_Parameter_Reduction_Percent":
                optuna_parameter_reduction,

            "Training_Time_Reduction_Percent":
                optuna_training_reduction,

            "Adapter_Size_MB":
                optuna[
                    "adapter_size_mb"
                ]
        }
    ]
)

# ============================================================
# SAVE
# ============================================================

ablation_df.to_csv(
    OUTPUT_DIR
    / "06_ablation_full_table.csv",
    index=False
)


effects_df.to_csv(
    OUTPUT_DIR
    / "06_component_effects.csv",
    index=False
)


efficiency_df.to_csv(
    OUTPUT_DIR
    / "06_efficiency_summary.csv",
    index=False
)

# ============================================================
# PAPER-FRIENDLY TABLE
# ============================================================

paper_df = (
    ablation_df[
        [
            "Experiment",
            "Model",
            "LoRA",
            "Optuna",
            "Threshold_Optimization",
            "Threshold",
            "Test_F1",
            "ROC_AUC",
            "PR_AUC",
            "Trainable_Percent"
        ]
    ]
    .copy()
)


paper_df[
    "Threshold"
] = (
    paper_df[
        "Threshold"
    ]
    .round(2)
)


for col in [
    "Test_F1",
    "ROC_AUC",
    "PR_AUC"
]:

    paper_df[col] = (
        paper_df[col]
        .round(4)
    )


paper_df[
    "Trainable_Percent"
] = (
    paper_df[
        "Trainable_Percent"
    ]
    .round(4)
)


paper_df.to_csv(
    OUTPUT_DIR
    / "06_ablation_paper_table.csv",
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 85)
print("ABLATION STUDY")
print("=" * 85)

print(
    paper_df.to_string(
        index=False
    )
)


print("\n" + "=" * 85)
print("COMPONENT EFFECTS")
print("=" * 85)

effects_print = (
    effects_df.copy()
)

effects_print[
    "Before_F1"
] = (
    effects_print[
        "Before_F1"
    ]
    .round(4)
)

effects_print[
    "After_F1"
] = (
    effects_print[
        "After_F1"
    ]
    .round(4)
)

effects_print[
    "Delta_F1"
] = (
    effects_print[
        "Delta_F1"
    ]
    .round(4)
)

print(
    effects_print.to_string(
        index=False
    )
)


print("\n" + "=" * 85)
print("PARAMETER / TRAINING EFFICIENCY")
print("=" * 85)

efficiency_print = (
    efficiency_df.copy()
)

efficiency_print[
    "Trainable_Parameter_Reduction_Percent"
] = (
    efficiency_print[
        "Trainable_Parameter_Reduction_Percent"
    ]
    .round(2)
)

efficiency_print[
    "Training_Time_Reduction_Percent"
] = (
    efficiency_print[
        "Training_Time_Reduction_Percent"
    ]
    .round(2)
)

efficiency_print[
    "Adapter_Size_MB"
] = (
    efficiency_print[
        "Adapter_Size_MB"
    ]
    .round(2)
)

print(
    efficiency_print.to_string(
        index=False
    )
)

# ============================================================
# SHORT INTERPRETATION
# ============================================================

print("\n" + "=" * 85)
print("KEY ABLATION FINDINGS")
print("=" * 85)

print(
    "DistilBERT F1 at default threshold:",
    round(
        distilbert_default_f1,
        4
    )
)

print(
    "Fixed LoRA F1 at default threshold:",
    round(
        lora_default_f1,
        4
    )
)

print(
    "Optuna-LoRA F1 at default threshold:",
    round(
        optuna_default_f1,
        4
    )
)

print(
    "Final Optuna-LoRA F1:",
    round(
        optuna_threshold_f1,
        4
    )
)

print(
    "\nLoRA trainable parameter reduction:",
    round(
        lora_parameter_reduction,
        2
    ),
    "%"
)

print(
    "Optuna-LoRA trainable parameter reduction:",
    round(
        optuna_parameter_reduction,
        2
    ),
    "%"
)

print(
    "\nResults folder:",
    OUTPUT_DIR
)

print(
    "\nSTEP 9 COMPLETED SUCCESSFULLY"
)