import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

warnings.filterwarnings("ignore")

# ============================================================
# STEP 5
# REPEATED CROSS-VALIDATION
# ============================================================

INPUT_FILE = "cicd_multiproject_leakage_safe.csv"
OUTPUT_FOLDER = "results_cross_validation"

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 3

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

print("=" * 75)
print("STEP 5 - REPEATED 5-FOLD CROSS-VALIDATION")
print("=" * 75)

# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

    print("XGBoost available: YES")

except Exception:

    XGBOOST_AVAILABLE = False

    print("XGBoost available: NO")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

required_columns = [
    "project_name",
    "log_text",
    "label"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


df = df.dropna(
    subset=[
        "log_text",
        "label"
    ]
).copy()

df["log_text"] = (
    df["log_text"]
    .astype(str)
)

df["label"] = (
    df["label"]
    .astype(int)
)

print("\nDataset rows:", len(df))

print(
    "Unique projects:",
    df["project_name"].nunique()
)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# DATA
# ============================================================

X_text = (
    df["log_text"]
    .values
)

y = (
    df["label"]
    .values
)


# ============================================================
# MODEL FACTORY
#
# Fresh model created for every fold.
# ============================================================

def create_models():

    models = {

        "Naive_Bayes":
            MultinomialNB(),

        "Logistic_Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE
            ),

        "Linear_SVM":
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Random_Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    }

    if XGBOOST_AVAILABLE:

        models[
            "XGBoost"
        ] = XGBClassifier(

            n_estimators=300,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.9,

            colsample_bytree=0.9,

            eval_metric="logloss",

            random_state=RANDOM_STATE,

            n_jobs=-1
        )

    return models


# ============================================================
# SCORE FUNCTION
# ============================================================

def get_scores(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        return (
            model
            .predict_proba(X)[:, 1]
        )

    if hasattr(
        model,
        "decision_function"
    ):

        raw_scores = (
            model
            .decision_function(X)
        )

        # Convert to probability-like scores
        return (
            1.0 /
            (
                1.0
                + np.exp(
                    -raw_scores
                )
            )
        )

    return (
        model.predict(X)
    )


# ============================================================
# VALIDATION-ONLY THRESHOLD SELECTION
# ============================================================

def find_best_threshold(
    y_true,
    scores
):

    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            scores
            >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


# ============================================================
# REPEATED STRATIFIED CV
# ============================================================

cv = RepeatedStratifiedKFold(

    n_splits=N_SPLITS,

    n_repeats=N_REPEATS,

    random_state=RANDOM_STATE
)


all_results = []

total_folds = (
    N_SPLITS
    * N_REPEATS
)

print(
    "\nTotal evaluations per model:",
    total_folds
)


# ============================================================
# OUTER CV
# ============================================================

for fold_number, (
    outer_train_idx,
    test_idx
) in enumerate(
    cv.split(
        X_text,
        y
    ),
    start=1
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"OUTER FOLD "
        f"{fold_number}/{total_folds}"
    )

    print(
        "=" * 75
    )

    outer_train_text = (
        X_text[
            outer_train_idx
        ]
    )

    outer_train_y = (
        y[
            outer_train_idx
        ]
    )

    test_text = (
        X_text[
            test_idx
        ]
    )

    test_y = (
        y[
            test_idx
        ]
    )


    # ========================================================
    # INTERNAL TRAIN / VALIDATION
    #
    # Threshold must NOT be selected from outer test fold.
    # ========================================================

    inner_train_text, val_text, \
    inner_train_y, val_y = train_test_split(

        outer_train_text,

        outer_train_y,

        test_size=0.20,

        random_state=(
            RANDOM_STATE
            + fold_number
        ),

        stratify=outer_train_y
    )


    # ========================================================
    # TF-IDF
    #
    # Fit only on inner training data.
    # ========================================================

    vectorizer = TfidfVectorizer(

        max_features=20000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True
    )


    X_inner_train = (
        vectorizer
        .fit_transform(
            inner_train_text
        )
    )

    X_val = (
        vectorizer
        .transform(
            val_text
        )
    )

    X_test = (
        vectorizer
        .transform(
            test_text
        )
    )


    # ========================================================
    # MODELS
    # ========================================================

    models = create_models()

    for model_name, model in models.items():

        print(
            f"  {model_name}"
        )

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_inner_train,
            inner_train_y
        )


        # ----------------------------------------------------
        # Tune threshold using validation only
        # ----------------------------------------------------

        val_scores = get_scores(
            model,
            X_val
        )

        best_threshold, validation_f1 = (
            find_best_threshold(
                val_y,
                val_scores
            )
        )


        # ----------------------------------------------------
        # Evaluate untouched outer test fold
        # ----------------------------------------------------

        test_scores = get_scores(
            model,
            X_test
        )

        test_predictions = (
            test_scores
            >= best_threshold
        ).astype(int)


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            test_y,
            test_predictions
        )

        precision = precision_score(
            test_y,
            test_predictions,
            zero_division=0
        )

        recall = recall_score(
            test_y,
            test_predictions,
            zero_division=0
        )

        f1 = f1_score(
            test_y,
            test_predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            test_y,
            test_scores
        )

        pr_auc = average_precision_score(
            test_y,
            test_scores
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # Repeat and fold numbers
        # ----------------------------------------------------

        repeat_number = (
            (fold_number - 1)
            // N_SPLITS
            + 1
        )

        fold_within_repeat = (
            (fold_number - 1)
            % N_SPLITS
            + 1
        )


        result = {

            "model":
                model_name,

            "repeat":
                repeat_number,

            "fold":
                fold_within_repeat,

            "evaluation_number":
                fold_number,

            "threshold":
                best_threshold,

            "validation_f1":
                validation_f1,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "runtime_sec":
                elapsed
        }

        all_results.append(
            result
        )

        print(
            f"    Threshold = "
            f"{best_threshold:.2f}"
        )

        print(
            f"    F1 = "
            f"{f1:.4f}"
        )

        print(
            f"    PR-AUC = "
            f"{pr_auc:.4f}"
        )


# ============================================================
# SAVE FOLD-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

fold_file = os.path.join(

    OUTPUT_FOLDER,

    "02_repeated_cv_fold_results.csv"
)

results_df.to_csv(
    fold_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metric_columns = [

    "accuracy",

    "precision",

    "recall",

    "f1_score",

    "roc_auc",

    "pr_auc"
]


summary_rows = []

for model_name in (
    results_df["model"]
    .unique()
):

    subset = results_df[
        results_df["model"]
        == model_name
    ]

    row = {
        "model":
            model_name,

        "n_evaluations":
            len(subset)
    }

    for metric in metric_columns:

        row[
            f"{metric}_mean"
        ] = (
            subset[
                metric
            ]
            .mean()
        )

        row[
            f"{metric}_std"
        ] = (
            subset[
                metric
            ]
            .std()
        )

    row[
        "threshold_mean"
    ] = (
        subset[
            "threshold"
        ]
        .mean()
    )

    row[
        "threshold_std"
    ] = (
        subset[
            "threshold"
        ]
        .std()
    )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df = (
    summary_df
    .sort_values(
        by="f1_score_mean",
        ascending=False
    )
)


summary_file = os.path.join(

    OUTPUT_FOLDER,

    "02_repeated_cv_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PAPER-FRIENDLY RESULT
# ============================================================

paper_rows = []

for _, row in summary_df.iterrows():

    paper_rows.append({

        "Model":
            row["model"],

        "Accuracy":
            f"{row['accuracy_mean']:.4f} "
            f"± "
            f"{row['accuracy_std']:.4f}",

        "Precision":
            f"{row['precision_mean']:.4f} "
            f"± "
            f"{row['precision_std']:.4f}",

        "Recall":
            f"{row['recall_mean']:.4f} "
            f"± "
            f"{row['recall_std']:.4f}",

        "F1":
            f"{row['f1_score_mean']:.4f} "
            f"± "
            f"{row['f1_score_std']:.4f}",

        "ROC_AUC":
            f"{row['roc_auc_mean']:.4f} "
            f"± "
            f"{row['roc_auc_std']:.4f}",

        "PR_AUC":
            f"{row['pr_auc_mean']:.4f} "
            f"± "
            f"{row['pr_auc_std']:.4f}"
    })


paper_df = pd.DataFrame(
    paper_rows
)


paper_file = os.path.join(

    OUTPUT_FOLDER,

    "02_repeated_cv_paper_table.csv"
)

paper_df.to_csv(
    paper_file,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "REPEATED CROSS-VALIDATION SUMMARY"
)

print(
    "=" * 75
)

print(
    paper_df.to_string(
        index=False
    )
)


print(
    "\nFold-level results:"
)

print(
    fold_file
)

print(
    "\nSummary:"
)

print(
    summary_file
)

print(
    "\nPaper-ready table:"
)

print(
    paper_file
)

print(
    "\nSTEP 5 COMPLETED SUCCESSFULLY"
)