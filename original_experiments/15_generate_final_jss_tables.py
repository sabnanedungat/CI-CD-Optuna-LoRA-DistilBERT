import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 17
# GENERATE FINAL JSS TABLES AND FIGURES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "FINAL_JSS_RESULTS"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 95)
print("STEP 17 - FINAL JSS RESULT CONSOLIDATION")
print("=" * 95)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):

    if not path.exists():

        print(
            f"WARNING - Missing file:\n{path}"
        )

        return None

    with open(
        path,
        "r"
    ) as file:

        return json.load(
            file
        )


def load_csv(path):

    if not path.exists():

        print(
            f"WARNING - Missing file:\n{path}"
        )

        return None

    return pd.read_csv(
        path
    )


def save_table(
    dataframe,
    filename
):

    path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        path,
        index=False
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# INPUT FILES
# ============================================================

BASELINE_FILE = (
    BASE_DIR
    / "results_baseline"
    / "01_baseline_all_results.csv"
)

CV_FILE = (
    BASE_DIR
    / "results_cross_validation"
    / "02_repeated_cv_summary.csv"
)

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

ABLATION_FILE = (
    BASE_DIR
    / "results_ablation"
    / "06_ablation_paper_table.csv"
)

STAT_FILE = (
    BASE_DIR
    / "results_statistics"
    / "07_statistical_paper_table.csv"
)

BOOTSTRAP_FILE = (
    BASE_DIR
    / "results_statistics"
    / "07_bootstrap_confidence_intervals.csv"
)

CROSS_DISTILBERT_FILE = (
    BASE_DIR
    / "results_cross_project"
    / "distilbert"
    / "08B_cross_project_distilbert_results.json"
)

CROSS_LORA_FILE = (
    BASE_DIR
    / "results_cross_project"
    / "final_optuna_lora"
    / "08F_final_results.json"
)

PARTIAL_FILE = (
    BASE_DIR
    / "results_partial_logs"
    / "09_partial_log_full_results.csv"
)

KAFKA_STEADY_FILE = (
    BASE_DIR
    / "results_kafka_steady"
    / "12_kafka_performance_summary.csv"
)

# fallback if user has not copied folder
if not KAFKA_STEADY_FILE.exists():

    KAFKA_STEADY_FILE = (
        BASE_DIR
        / "results_kafka_performance"
        / "12_kafka_performance_summary.csv"
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

baseline = load_csv(
    BASELINE_FILE
)

cv = load_csv(
    CV_FILE
)

dist = load_json(
    DISTILBERT_FILE
)

lora = load_json(
    LORA_FILE
)

optuna = load_json(
    OPTUNA_FILE
)

ablation = load_csv(
    ABLATION_FILE
)

statistics = load_csv(
    STAT_FILE
)

bootstrap = load_csv(
    BOOTSTRAP_FILE
)

cross_dist = load_json(
    CROSS_DISTILBERT_FILE
)

cross_lora = load_json(
    CROSS_LORA_FILE
)

partial = load_csv(
    PARTIAL_FILE
)

kafka_steady = load_csv(
    KAFKA_STEADY_FILE
)


# ============================================================
# TABLE 1
# CLASSICAL BASELINES - FULL LOG
# ============================================================

print(
    "\nCreating Table 1..."
)

if baseline is not None:

    full_baseline = baseline[
        baseline[
            "log_availability"
        ].astype(str)
        .isin(
            [
                "100%",
                "100"
            ]
        )
    ].copy()

    if len(
        full_baseline
    ) == 0:

        # fallback
        full_baseline = baseline.copy()

        if (
            "log_availability"
            in full_baseline.columns
        ):

            full_baseline = full_baseline[
                full_baseline[
                    "log_availability"
                ]
                == full_baseline[
                    "log_availability"
                ].max()
            ]

    baseline_columns = [

        column

        for column in [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "training_time_sec",
            "prediction_time_per_build_ms"
        ]

        if column
        in full_baseline.columns
    ]

    table1 = full_baseline[
        baseline_columns
    ].copy()

    numeric_cols = table1.select_dtypes(
        include=np.number
    ).columns

    table1[
        numeric_cols
    ] = table1[
        numeric_cols
    ].round(4)

    save_table(
        table1,
        "Table_1_Classical_Baselines.csv"
    )


# ============================================================
# TABLE 2
# REPEATED CROSS VALIDATION
# ============================================================

print(
    "\nCreating Table 2..."
)

if cv is not None:

    table2 = cv.copy()

    numeric_cols = table2.select_dtypes(
        include=np.number
    ).columns

    table2[
        numeric_cols
    ] = table2[
        numeric_cols
    ].round(4)

    save_table(
        table2,
        "Table_2_Repeated_Cross_Validation.csv"
    )


# ============================================================
# TABLE 3
# DISTILBERT / LoRA / OPTUNA-LoRA
# ============================================================

print(
    "\nCreating Table 3..."
)

transformer_rows = []

if dist is not None:

    transformer_rows.append(
        {
            "Model":
                "DistilBERT",

            "Trainable Parameters":
                dist[
                    "trainable_parameters"
                ],

            "Trainable %":
                dist[
                    "trainable_percentage"
                ],

            "Threshold":
                dist[
                    "optimized_threshold"
                ],

            "Accuracy":
                dist[
                    "test_accuracy"
                ],

            "Precision":
                dist[
                    "test_precision"
                ],

            "Recall":
                dist[
                    "test_recall"
                ],

            "F1":
                dist[
                    "test_f1"
                ],

            "ROC-AUC":
                dist[
                    "test_roc_auc"
                ],

            "PR-AUC":
                dist[
                    "test_pr_auc"
                ],

            "Training Time (s)":
                dist[
                    "training_time_sec"
                ],

            "Mean Latency (ms)":
                dist[
                    "mean_latency_ms"
                ]
        }
    )


if lora is not None:

    transformer_rows.append(
        {
            "Model":
                "LoRA-DistilBERT",

            "Trainable Parameters":
                lora[
                    "trainable_parameters"
                ],

            "Trainable %":
                lora[
                    "trainable_percentage"
                ],

            "Threshold":
                lora[
                    "threshold"
                ],

            "Accuracy":
                lora[
                    "accuracy"
                ],

            "Precision":
                lora[
                    "precision"
                ],

            "Recall":
                lora[
                    "recall"
                ],

            "F1":
                lora[
                    "test_f1"
                ],

            "ROC-AUC":
                lora[
                    "roc_auc"
                ],

            "PR-AUC":
                lora[
                    "pr_auc"
                ],

            "Training Time (s)":
                lora[
                    "training_time_sec"
                ],

            "Mean Latency (ms)":
                lora[
                    "mean_latency_ms"
                ]
        }
    )


if optuna is not None:

    transformer_rows.append(
        {
            "Model":
                "Optuna-LoRA-DistilBERT",

            "Trainable Parameters":
                optuna[
                    "trainable_parameters"
                ],

            "Trainable %":
                optuna[
                    "trainable_percentage"
                ],

            "Threshold":
                optuna[
                    "optimized_threshold"
                ],

            "Accuracy":
                optuna[
                    "test_accuracy"
                ],

            "Precision":
                optuna[
                    "test_precision"
                ],

            "Recall":
                optuna[
                    "test_recall"
                ],

            "F1":
                optuna[
                    "test_f1"
                ],

            "ROC-AUC":
                optuna[
                    "test_roc_auc"
                ],

            "PR-AUC":
                optuna[
                    "test_pr_auc"
                ],

            "Training Time (s)":
                optuna[
                    "final_training_time_sec"
                ],

            "Mean Latency (ms)":
                optuna[
                    "mean_latency_ms"
                ]
        }
    )


table3 = pd.DataFrame(
    transformer_rows
)

numeric_cols = table3.select_dtypes(
    include=np.number
).columns

table3[
    numeric_cols
] = table3[
    numeric_cols
].round(4)

save_table(
    table3,
    "Table_3_Transformer_Comparison.csv"
)


# ============================================================
# TABLE 4
# ABLATION
# ============================================================

print(
    "\nCreating Table 4..."
)

if ablation is not None:

    save_table(
        ablation,
        "Table_4_Ablation_Study.csv"
    )


# ============================================================
# TABLE 5
# STATISTICAL SIGNIFICANCE
# ============================================================

print(
    "\nCreating Table 5..."
)

if statistics is not None:

    save_table(
        statistics,
        "Table_5_Statistical_Significance.csv"
    )


if bootstrap is not None:

    save_table(
        bootstrap,
        "Table_5B_Bootstrap_Confidence_Intervals.csv"
    )


# ============================================================
# TABLE 6
# CROSS-PROJECT GENERALIZATION
# ============================================================

print(
    "\nCreating Table 6..."
)

cross_rows = []


if cross_dist is not None:

    cross_rows.append(
        {
            "Model":
                "DistilBERT",

            "Accuracy":
                cross_dist[
                    "test_accuracy"
                ],

            "Precision":
                cross_dist[
                    "test_precision"
                ],

            "Recall":
                cross_dist[
                    "test_recall"
                ],

            "F1":
                cross_dist[
                    "test_f1"
                ],

            "ROC-AUC":
                cross_dist[
                    "test_roc_auc"
                ],

            "PR-AUC":
                cross_dist[
                    "test_pr_auc"
                ],

            "Trainable Parameters":
                cross_dist[
                    "trainable_parameters"
                ]
        }
    )


if cross_lora is not None:

    main = cross_lora[
        "main_f1_operating_point"
    ]

    cross_rows.append(
        {
            "Model":
                "Optuna-LoRA-DistilBERT",

            "Accuracy":
                main[
                    "accuracy"
                ],

            "Precision":
                main[
                    "precision"
                ],

            "Recall":
                main[
                    "recall"
                ],

            "F1":
                main[
                    "f1"
                ],

            "ROC-AUC":
                main[
                    "roc_auc"
                ],

            "PR-AUC":
                main[
                    "pr_auc"
                ],

            "Trainable Parameters":
                cross_lora[
                    "trainable_parameters"
                ]
        }
    )


table6 = pd.DataFrame(
    cross_rows
)

numeric_cols = table6.select_dtypes(
    include=np.number
).columns

table6[
    numeric_cols
] = table6[
    numeric_cols
].round(4)

save_table(
    table6,
    "Table_6_Cross_Project_Generalization.csv"
)


# ============================================================
# TABLE 7
# PARTIAL LOG PERFORMANCE
# ============================================================

print(
    "\nCreating Table 7..."
)

if partial is not None:

    partial_columns = [

        column

        for column in [
            "log_availability",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp"
        ]

        if column
        in partial.columns
    ]

    table7 = partial[
        partial_columns
    ].copy()

    numeric_cols = table7.select_dtypes(
        include=np.number
    ).columns

    table7[
        numeric_cols
    ] = table7[
        numeric_cols
    ].round(4)

    save_table(
        table7,
        "Table_7_Partial_Log_Performance.csv"
    )


# ============================================================
# TABLE 8
# KAFKA DEPLOYMENT PERFORMANCE
# ============================================================

print(
    "\nCreating Table 8..."
)

kafka_rows = []


# ------------------------------------------------------------
# Burst benchmark
# Values preserved from completed experimental run.
# ------------------------------------------------------------

kafka_rows.append(
    {
        "Scenario":
            "Burst / Stress",

        "Messages":
            200,

        "Builds":
            50,

        "Producer Rate (msg/s)":
            763.594,

        "Consumer Rate (msg/s)":
            20.107,

        "Mean Inference (ms)":
            48.359,

        "Median Inference (ms)":
            43.600,

        "P95 Inference (ms)":
            68.839,

        "P99 Inference (ms)":
            118.126,

        "Mean E2E (ms)":
            5370.248,

        "Median E2E (ms)":
            5195.660,

        "P95 E2E (ms)":
            9569.713,

        "P99 E2E (ms)":
            9901.455
    }
)


# ------------------------------------------------------------
# Steady benchmark
# ------------------------------------------------------------

if kafka_steady is not None:

    k = kafka_steady.iloc[
        0
    ]

    kafka_rows.append(
        {
            "Scenario":
                "Steady State",

            "Messages":
                int(
                    k[
                        "messages_processed"
                    ]
                ),

            "Builds":
                int(
                    k[
                        "unique_builds"
                    ]
                ),

            "Producer Rate (msg/s)":
                9.342,

            "Consumer Rate (msg/s)":
                k[
                    "throughput_messages_per_sec"
                ],

            "Mean Inference (ms)":
                k[
                    "inference_mean_ms"
                ],

            "Median Inference (ms)":
                k[
                    "inference_median_ms"
                ],

            "P95 Inference (ms)":
                k[
                    "inference_p95_ms"
                ],

            "P99 Inference (ms)":
                k[
                    "inference_p99_ms"
                ],

            "Mean E2E (ms)":
                k[
                    "e2e_mean_ms"
                ],

            "Median E2E (ms)":
                k[
                    "e2e_median_ms"
                ],

            "P95 E2E (ms)":
                k[
                    "e2e_p95_ms"
                ],

            "P99 E2E (ms)":
                k[
                    "e2e_p99_ms"
                ]
        }
    )


table8 = pd.DataFrame(
    kafka_rows
)

numeric_cols = table8.select_dtypes(
    include=np.number
).columns

table8[
    numeric_cols
] = table8[
    numeric_cols
].round(3)

save_table(
    table8,
    "Table_8_Kafka_Deployment_Performance.csv"
)


# ============================================================
# MASTER KEY RESULTS TABLE
# ============================================================

print(
    "\nCreating master results..."
)

master_rows = []


if optuna is not None:

    master_rows.append(
        {
            "Experiment":
                "Random-split Optuna-LoRA",

            "Accuracy":
                optuna[
                    "test_accuracy"
                ],

            "Precision":
                optuna[
                    "test_precision"
                ],

            "Recall":
                optuna[
                    "test_recall"
                ],

            "F1":
                optuna[
                    "test_f1"
                ],

            "ROC-AUC":
                optuna[
                    "test_roc_auc"
                ],

            "PR-AUC":
                optuna[
                    "test_pr_auc"
                ]
        }
    )


if cross_lora is not None:

    main = cross_lora[
        "main_f1_operating_point"
    ]

    master_rows.append(
        {
            "Experiment":
                "Unseen-project Optuna-LoRA",

            "Accuracy":
                main[
                    "accuracy"
                ],

            "Precision":
                main[
                    "precision"
                ],

            "Recall":
                main[
                    "recall"
                ],

            "F1":
                main[
                    "f1"
                ],

            "ROC-AUC":
                main[
                    "roc_auc"
                ],

            "PR-AUC":
                main[
                    "pr_auc"
                ]
        }
    )


if partial is not None:

    partial_50 = partial[
        partial[
            "log_availability"
        ]
        == "50%"
    ]

    if len(
        partial_50
    ) > 0:

        p = partial_50.iloc[
            0
        ]

        master_rows.append(
            {
                "Experiment":
                    "50% Partial Log",

                "Accuracy":
                    p[
                        "accuracy"
                    ],

                "Precision":
                    p[
                        "precision"
                    ],

                "Recall":
                    p[
                        "recall"
                    ],

                "F1":
                    p[
                        "f1"
                    ],

                "ROC-AUC":
                    p[
                        "roc_auc"
                    ],

                "PR-AUC":
                    p[
                        "pr_auc"
                    ]
            }
        )


master_df = pd.DataFrame(
    master_rows
)

numeric_cols = master_df.select_dtypes(
    include=np.number
).columns

master_df[
    numeric_cols
] = master_df[
    numeric_cols
].round(4)

save_table(
    master_df,
    "MASTER_Key_Predictive_Results.csv"
)


# ============================================================
# FIGURE 1
# TRANSFORMER PERFORMANCE
# ============================================================

if len(
    table3
) > 0:

    figure_data = table3[
        [
            "Model",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].copy()

    figure_data = (
        figure_data
        .set_index(
            "Model"
        )
    )

    ax = figure_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Predictive Performance of Transformer Variants"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure_1_Transformer_Performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 2
# TRAINABLE PARAMETERS
# ============================================================

if len(
    table3
) > 0:

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        table3[
            "Model"
        ],
        table3[
            "Trainable %"
        ]
    )

    plt.ylabel(
        "Trainable Parameters (%)"
    )

    plt.title(
        "Parameter Efficiency of DistilBERT and LoRA Variants"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure_2_Parameter_Efficiency.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 3
# PARTIAL LOG PERFORMANCE
# ============================================================

if partial is not None:

    plot_df = (
        partial.copy()
    )

    order = [
        "25%",
        "50%",
        "75%",
        "100%"
    ]

    plot_df[
        "log_availability"
    ] = pd.Categorical(
        plot_df[
            "log_availability"
        ],
        categories=order,
        ordered=True
    )

    plot_df = (
        plot_df
        .sort_values(
            "log_availability"
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        plot_df[
            "log_availability"
        ].astype(str),
        plot_df[
            "f1"
        ],
        marker="o",
        label="F1"
    )

    plt.plot(
        plot_df[
            "log_availability"
        ].astype(str),
        plot_df[
            "roc_auc"
        ],
        marker="o",
        label="ROC-AUC"
    )

    plt.plot(
        plot_df[
            "log_availability"
        ].astype(str),
        plot_df[
            "pr_auc"
        ],
        marker="o",
        label="PR-AUC"
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Available CI Information"
    )

    plt.ylabel(
        "Performance"
    )

    plt.title(
        "Optuna-LoRA Performance Under Partial Information"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure_3_Partial_Log_Performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 4
# CROSS PROJECT GENERALIZATION
# ============================================================

if len(
    table6
) > 0:

    figure_data = (
        table6[
            [
                "Model",
                "F1",
                "ROC-AUC",
                "PR-AUC"
            ]
        ]
        .set_index(
            "Model"
        )
    )

    ax = figure_data.plot(
        kind="bar",
        figsize=(9, 6)
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Generalization to Completely Unseen Projects"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.xticks(
        rotation=10
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure_4_Cross_Project_Generalization.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 5
# KAFKA STEADY LATENCY
# ============================================================

if kafka_steady is not None:

    k = kafka_steady.iloc[
        0
    ]

    latency_labels = [
        "Mean",
        "Median",
        "P95",
        "P99"
    ]

    latency_values = [
        k[
            "e2e_mean_ms"
        ],
        k[
            "e2e_median_ms"
        ],
        k[
            "e2e_p95_ms"
        ],
        k[
            "e2e_p99_ms"
        ]
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        latency_labels,
        latency_values
    )

    plt.ylabel(
        "End-to-End Latency (ms)"
    )

    plt.title(
        "Kafka Steady-State End-to-End Latency"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure_5_Kafka_Steady_State_Latency.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "FINAL JSS EXPERIMENTAL RESULTS"
)

summary_lines.append(
    "=" * 70
)


if optuna is not None:

    summary_lines.append(
        "\nRandom-split Optuna-LoRA:"
    )

    summary_lines.append(
        f"Accuracy = "
        f"{optuna['test_accuracy']:.4f}"
    )

    summary_lines.append(
        f"F1 = "
        f"{optuna['test_f1']:.4f}"
    )

    summary_lines.append(
        f"ROC-AUC = "
        f"{optuna['test_roc_auc']:.4f}"
    )

    summary_lines.append(
        f"PR-AUC = "
        f"{optuna['test_pr_auc']:.4f}"
    )

    summary_lines.append(
        f"Trainable parameters = "
        f"{optuna['trainable_percentage']:.4f}%"
    )


if cross_lora is not None:

    main = cross_lora[
        "main_f1_operating_point"
    ]

    summary_lines.append(
        "\nCross-project Optuna-LoRA:"
    )

    summary_lines.append(
        f"Accuracy = "
        f"{main['accuracy']:.4f}"
    )

    summary_lines.append(
        f"F1 = "
        f"{main['f1']:.4f}"
    )

    summary_lines.append(
        f"ROC-AUC = "
        f"{main['roc_auc']:.4f}"
    )

    summary_lines.append(
        f"PR-AUC = "
        f"{main['pr_auc']:.4f}"
    )


if partial is not None:

    partial_50 = partial[
        partial[
            "log_availability"
        ]
        == "50%"
    ]

    if len(
        partial_50
    ) > 0:

        row = partial_50.iloc[
            0
        ]

        summary_lines.append(
            "\n50% partial-log prediction:"
        )

        summary_lines.append(
            f"Recall = "
            f"{row['recall']:.4f}"
        )

        summary_lines.append(
            f"F1 = "
            f"{row['f1']:.4f}"
        )

        summary_lines.append(
            f"PR-AUC = "
            f"{row['pr_auc']:.4f}"
        )


if kafka_steady is not None:

    k = kafka_steady.iloc[
        0
    ]

    summary_lines.append(
        "\nKafka steady-state deployment:"
    )

    summary_lines.append(
        f"Throughput = "
        f"{k['throughput_messages_per_sec']:.3f} "
        f"messages/sec"
    )

    summary_lines.append(
        f"Mean inference latency = "
        f"{k['inference_mean_ms']:.3f} ms"
    )

    summary_lines.append(
        f"Mean E2E latency = "
        f"{k['e2e_mean_ms']:.3f} ms"
    )

    summary_lines.append(
        f"P95 E2E latency = "
        f"{k['e2e_p95_ms']:.3f} ms"
    )


summary_file = (
    OUTPUT_DIR
    / "FINAL_JSS_RESULT_SUMMARY.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# FINAL DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "FINAL JSS FILES CREATED"
)

print(
    "=" * 95
)

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        file.name
    )


print(
    "\nFinal results folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSTEP 17 COMPLETED SUCCESSFULLY"
)