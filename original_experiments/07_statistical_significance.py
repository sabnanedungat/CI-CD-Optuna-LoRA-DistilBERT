import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# STEP 10
# STATISTICAL SIGNIFICANCE ANALYSIS
# ============================================================

SEED = 42
N_BOOTSTRAP = 5000
ALPHA = 0.05

BASE_DIR = Path(__file__).resolve().parent

DISTILBERT_PRED_FILE = (
    BASE_DIR
    / "results_distilbert"
    / "03_distilbert_test_predictions.csv"
)

OPTUNA_PRED_FILE = (
    BASE_DIR
    / "results_optuna_lora"
    / "05_optuna_lora_test_predictions.csv"
)

DISTILBERT_RESULT_FILE = (
    BASE_DIR
    / "results_distilbert"
    / "03_distilbert_final_results.json"
)

OPTUNA_RESULT_FILE = (
    BASE_DIR
    / "results_optuna_lora"
    / "05_optuna_lora_final_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results_statistics"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

print("=" * 85)
print("STEP 10 - STATISTICAL SIGNIFICANCE TESTING")
print("=" * 85)

# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DISTILBERT_PRED_FILE,
    OPTUNA_PRED_FILE,
    DISTILBERT_RESULT_FILE,
    OPTUNA_RESULT_FILE
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

# ============================================================
# LOAD PREDICTIONS
# ============================================================

dist = pd.read_csv(
    DISTILBERT_PRED_FILE
)

optuna = pd.read_csv(
    OPTUNA_PRED_FILE
)

with open(
    DISTILBERT_RESULT_FILE,
    "r"
) as file:

    dist_result = json.load(
        file
    )

with open(
    OPTUNA_RESULT_FILE,
    "r"
) as file:

    optuna_result = json.load(
        file
    )

# ============================================================
# VERIFY TEST SET ALIGNMENT
# ============================================================

if len(dist) != len(optuna):

    raise ValueError(
        "DistilBERT and Optuna-LoRA prediction files "
        "contain different numbers of test samples."
    )

dist_true = (
    dist["true_label"]
    .astype(int)
    .to_numpy()
)

optuna_true = (
    optuna["true_label"]
    .astype(int)
    .to_numpy()
)

if not np.array_equal(
    dist_true,
    optuna_true
):

    raise ValueError(
        "The two prediction files are not aligned "
        "to the same test examples."
    )

y_true = dist_true

dist_prob = (
    dist["failure_probability"]
    .astype(float)
    .to_numpy()
)

optuna_prob = (
    optuna["failure_probability"]
    .astype(float)
    .to_numpy()
)

print(
    "\nAligned test samples:",
    len(y_true)
)

# ============================================================
# THRESHOLDS
# ============================================================

DIST_DEFAULT_THRESHOLD = 0.50

DIST_OPT_THRESHOLD = float(
    dist_result[
        "optimized_threshold"
    ]
)

OPTUNA_DEFAULT_THRESHOLD = 0.50

OPTUNA_FINAL_THRESHOLD = float(
    optuna_result[
        "optimized_threshold"
    ]
)

print(
    "\nDistilBERT optimized threshold:",
    DIST_OPT_THRESHOLD
)

print(
    "Optuna-LoRA optimized threshold:",
    OPTUNA_FINAL_THRESHOLD
)

# ============================================================
# CREATE PREDICTIONS
# ============================================================

dist_default_pred = (
    dist_prob
    >= DIST_DEFAULT_THRESHOLD
).astype(int)

dist_optimized_pred = (
    dist_prob
    >= DIST_OPT_THRESHOLD
).astype(int)

optuna_default_pred = (
    optuna_prob
    >= OPTUNA_DEFAULT_THRESHOLD
).astype(int)

optuna_final_pred = (
    optuna_prob
    >= OPTUNA_FINAL_THRESHOLD
).astype(int)

# ============================================================
# MCNEMAR EXACT TEST
#
# This assesses whether the two paired classifiers have
# significantly different error patterns.
# ============================================================

def mcnemar_exact(
    y,
    prediction_a,
    prediction_b
):

    correct_a = (
        prediction_a
        == y
    )

    correct_b = (
        prediction_b
        == y
    )

    # A correct, B wrong
    b = int(
        np.sum(
            correct_a
            & ~correct_b
        )
    )

    # A wrong, B correct
    c = int(
        np.sum(
            ~correct_a
            & correct_b
        )
    )

    discordant = (
        b + c
    )

    if discordant == 0:

        p_value = 1.0

    else:

        result = binomtest(
            b,
            n=discordant,
            p=0.5,
            alternative="two-sided"
        )

        p_value = float(
            result.pvalue
        )

    return (
        b,
        c,
        discordant,
        p_value
    )

# ============================================================
# PAIRED BOOTSTRAP
#
# Gives 95% confidence interval for differences in:
# F1
# ROC-AUC
# PR-AUC
# ============================================================

def paired_bootstrap(
    y,
    probabilities_a,
    probabilities_b,
    predictions_a,
    predictions_b,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED
):

    rng = np.random.default_rng(
        seed
    )

    n = len(y)

    f1_differences = []
    roc_differences = []
    pr_differences = []

    for _ in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_sample = (
            y[
                indices
            ]
        )

        pred_a_sample = (
            predictions_a[
                indices
            ]
        )

        pred_b_sample = (
            predictions_b[
                indices
            ]
        )

        prob_a_sample = (
            probabilities_a[
                indices
            ]
        )

        prob_b_sample = (
            probabilities_b[
                indices
            ]
        )

        # B - A
        f1_a = f1_score(
            y_sample,
            pred_a_sample,
            zero_division=0
        )

        f1_b = f1_score(
            y_sample,
            pred_b_sample,
            zero_division=0
        )

        f1_differences.append(
            f1_b - f1_a
        )

        # ROC-AUC needs both classes
        if len(
            np.unique(
                y_sample
            )
        ) == 2:

            roc_a = roc_auc_score(
                y_sample,
                prob_a_sample
            )

            roc_b = roc_auc_score(
                y_sample,
                prob_b_sample
            )

            roc_differences.append(
                roc_b - roc_a
            )

            pr_a = average_precision_score(
                y_sample,
                prob_a_sample
            )

            pr_b = average_precision_score(
                y_sample,
                prob_b_sample
            )

            pr_differences.append(
                pr_b - pr_a
            )

    def summarize(
        values
    ):

        values = np.asarray(
            values
        )

        return {
            "mean_difference":
                float(
                    np.mean(
                        values
                    )
                ),

            "ci_lower":
                float(
                    np.percentile(
                        values,
                        2.5
                    )
                ),

            "ci_upper":
                float(
                    np.percentile(
                        values,
                        97.5
                    )
                )
        }

    return {
        "F1":
            summarize(
                f1_differences
            ),

        "ROC_AUC":
            summarize(
                roc_differences
            ),

        "PR_AUC":
            summarize(
                pr_differences
            )
    }

# ============================================================
# COMPARISONS
# ============================================================

comparisons = [
    {
        "name":
            "DistilBERT default vs Optuna-LoRA final",

        "a_name":
            "DistilBERT_0.50",

        "b_name":
            "Optuna_LoRA_optimized",

        "pred_a":
            dist_default_pred,

        "pred_b":
            optuna_final_pred,

        "prob_a":
            dist_prob,

        "prob_b":
            optuna_prob
    },

    {
        "name":
            "DistilBERT optimized vs Optuna-LoRA final",

        "a_name":
            "DistilBERT_optimized",

        "b_name":
            "Optuna_LoRA_optimized",

        "pred_a":
            dist_optimized_pred,

        "pred_b":
            optuna_final_pred,

        "prob_a":
            dist_prob,

        "prob_b":
            optuna_prob
    },

    {
        "name":
            "Optuna-LoRA default vs optimized threshold",

        "a_name":
            "Optuna_LoRA_0.50",

        "b_name":
            "Optuna_LoRA_optimized",

        "pred_a":
            optuna_default_pred,

        "pred_b":
            optuna_final_pred,

        "prob_a":
            optuna_prob,

        "prob_b":
            optuna_prob
    }
]

statistical_rows = []
bootstrap_rows = []

# ============================================================
# RUN TESTS
# ============================================================

for comparison in comparisons:

    print(
        "\n"
        + "=" * 85
    )

    print(
        comparison[
            "name"
        ]
    )

    print(
        "=" * 85
    )

    pred_a = comparison[
        "pred_a"
    ]

    pred_b = comparison[
        "pred_b"
    ]

    prob_a = comparison[
        "prob_a"
    ]

    prob_b = comparison[
        "prob_b"
    ]

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy_a = accuracy_score(
        y_true,
        pred_a
    )

    accuracy_b = accuracy_score(
        y_true,
        pred_b
    )

    f1_a = f1_score(
        y_true,
        pred_a,
        zero_division=0
    )

    f1_b = f1_score(
        y_true,
        pred_b,
        zero_division=0
    )

    roc_a = roc_auc_score(
        y_true,
        prob_a
    )

    roc_b = roc_auc_score(
        y_true,
        prob_b
    )

    pr_a = average_precision_score(
        y_true,
        prob_a
    )

    pr_b = average_precision_score(
        y_true,
        prob_b
    )

    # --------------------------------------------------------
    # McNemar
    # --------------------------------------------------------

    b_count, \
    c_count, \
    discordant, \
    p_value = mcnemar_exact(
        y_true,
        pred_a,
        pred_b
    )

    significant = (
        p_value
        < ALPHA
    )

    print(
        comparison[
            "a_name"
        ],
        "F1 =",
        round(
            f1_a,
            4
        )
    )

    print(
        comparison[
            "b_name"
        ],
        "F1 =",
        round(
            f1_b,
            4
        )
    )

    print(
        "F1 difference (B - A) =",
        round(
            f1_b - f1_a,
            4
        )
    )

    print(
        "\nMcNemar discordant pairs:"
    )

    print(
        "A correct / B wrong:",
        b_count
    )

    print(
        "A wrong / B correct:",
        c_count
    )

    print(
        "McNemar p-value:",
        round(
            p_value,
            6
        )
    )

    print(
        "Statistically significant at 0.05:",
        significant
    )

    statistical_rows.append(
        {
            "Comparison":
                comparison[
                    "name"
                ],

            "Model_A":
                comparison[
                    "a_name"
                ],

            "Model_B":
                comparison[
                    "b_name"
                ],

            "Accuracy_A":
                accuracy_a,

            "Accuracy_B":
                accuracy_b,

            "F1_A":
                f1_a,

            "F1_B":
                f1_b,

            "F1_Difference_B_minus_A":
                f1_b - f1_a,

            "ROC_AUC_A":
                roc_a,

            "ROC_AUC_B":
                roc_b,

            "PR_AUC_A":
                pr_a,

            "PR_AUC_B":
                pr_b,

            "A_correct_B_wrong":
                b_count,

            "A_wrong_B_correct":
                c_count,

            "Discordant_pairs":
                discordant,

            "McNemar_p_value":
                p_value,

            "Significant_at_0.05":
                significant
        }
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap = paired_bootstrap(
        y_true,
        prob_a,
        prob_b,
        pred_a,
        pred_b
    )

    print(
        "\nBootstrap F1 difference:"
    )

    print(
        "Mean difference:",
        round(
            bootstrap[
                "F1"
            ][
                "mean_difference"
            ],
            4
        )
    )

    print(
        "95% CI:",
        round(
            bootstrap[
                "F1"
            ][
                "ci_lower"
            ],
            4
        ),
        "to",
        round(
            bootstrap[
                "F1"
            ][
                "ci_upper"
            ],
            4
        )
    )

    for metric_name in [
        "F1",
        "ROC_AUC",
        "PR_AUC"
    ]:

        result = bootstrap[
            metric_name
        ]

        bootstrap_rows.append(
            {
                "Comparison":
                    comparison[
                        "name"
                    ],

                "Metric":
                    metric_name,

                "Mean_Difference_B_minus_A":
                    result[
                        "mean_difference"
                    ],

                "CI_95_Lower":
                    result[
                        "ci_lower"
                    ],

                "CI_95_Upper":
                    result[
                        "ci_upper"
                    ],

                "CI_Includes_Zero":
                    (
                        result[
                            "ci_lower"
                        ]
                        <= 0
                        <= result[
                            "ci_upper"
                        ]
                    )
            }
        )

# ============================================================
# SAVE RESULTS
# ============================================================

statistics_df = pd.DataFrame(
    statistical_rows
)

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

statistics_df.to_csv(
    OUTPUT_DIR
    / "07_mcnemar_results.csv",
    index=False
)

bootstrap_df.to_csv(
    OUTPUT_DIR
    / "07_bootstrap_confidence_intervals.csv",
    index=False
)

# ============================================================
# PAPER-FRIENDLY MCNEMAR TABLE
# ============================================================

paper_statistics = (
    statistics_df[
        [
            "Comparison",
            "F1_A",
            "F1_B",
            "F1_Difference_B_minus_A",
            "McNemar_p_value",
            "Significant_at_0.05"
        ]
    ]
    .copy()
)

for column in [
    "F1_A",
    "F1_B",
    "F1_Difference_B_minus_A",
    "McNemar_p_value"
]:

    paper_statistics[
        column
    ] = (
        paper_statistics[
            column
        ]
        .round(4)
    )

paper_statistics.to_csv(
    OUTPUT_DIR
    / "07_statistical_paper_table.csv",
    index=False
)

# ============================================================
# DISPLAY FINAL TABLES
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "STATISTICAL SIGNIFICANCE SUMMARY"
)

print(
    "=" * 85
)

print(
    paper_statistics.to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 85
)

print(
    "BOOTSTRAP CONFIDENCE INTERVALS"
)

print(
    "=" * 85
)

bootstrap_print = (
    bootstrap_df.copy()
)

for column in [
    "Mean_Difference_B_minus_A",
    "CI_95_Lower",
    "CI_95_Upper"
]:

    bootstrap_print[
        column
    ] = (
        bootstrap_print[
            column
        ]
        .round(4)
    )

print(
    bootstrap_print.to_string(
        index=False
    )
)

# ============================================================
# MAIN INTERPRETATION
# ============================================================

main_row = (
    statistics_df.iloc[0]
)

main_bootstrap = (
    bootstrap_df[
        (
            bootstrap_df[
                "Comparison"
            ]
            ==
            "DistilBERT default vs Optuna-LoRA final"
        )
        &
        (
            bootstrap_df[
                "Metric"
            ]
            == "F1"
        )
    ]
    .iloc[0]
)

print(
    "\n"
    + "=" * 85
)

print(
    "MAIN JSS INTERPRETATION"
)

print(
    "=" * 85
)

print(
    "DistilBERT default-threshold F1:",
    round(
        main_row[
            "F1_A"
        ],
        4
    )
)

print(
    "Final Optuna-LoRA F1:",
    round(
        main_row[
            "F1_B"
        ],
        4
    )
)

print(
    "Difference:",
    round(
        main_row[
            "F1_Difference_B_minus_A"
        ],
        4
    )
)

print(
    "McNemar p-value:",
    round(
        main_row[
            "McNemar_p_value"
        ],
        6
    )
)

print(
    "F1 bootstrap 95% CI:",
    round(
        main_bootstrap[
            "CI_95_Lower"
        ],
        4
    ),
    "to",
    round(
        main_bootstrap[
            "CI_95_Upper"
        ],
        4
    )
)

if (
    main_row[
        "Significant_at_0.05"
    ]
):

    print(
        "\nThe prediction difference is statistically significant."
    )

else:

    print(
        "\nThe prediction difference is NOT statistically significant."
    )

    print(
        "Therefore, the paper should emphasize that "
        "Optuna-LoRA maintains comparable predictive performance "
        "while greatly reducing trainable parameters, rather than "
        "claiming a major predictive improvement."
    )

print(
    "\nResults folder:",
    OUTPUT_DIR
)

print(
    "\nSTEP 10 COMPLETED SUCCESSFULLY"
)