import os
import re
import time
import json
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# STEP 4
# LEAKAGE-SAFE MULTI-PROJECT BASELINE MODELS
# ============================================================

INPUT_FILE = "cicd_multiproject_leakage_safe.csv"
OUTPUT_FOLDER = "results_baseline"

RANDOM_STATE = 42

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

print("=" * 70)
print("STEP 4 - LEAKAGE-SAFE BASELINE MODEL EVALUATION")
print("=" * 70)


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

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

df = df.dropna(
    subset=[
        "project_name",
        "log_text",
        "label"
    ]
).copy()

df["label"] = (
    df["label"]
    .astype(int)
)

df["log_text"] = (
    df["log_text"]
    .astype(str)
)

print("\nDataset loaded successfully")

print(
    "Total rows:",
    len(df)
)

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
# REMOVE PROJECT NAME FROM MODEL INPUT
#
# project_name remains in the dataframe for later
# cross-project evaluation, but the model should not simply
# memorize a project's identity.
# ============================================================

def clean_project_name(project):

    project = str(project).lower()

    project = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        project
    )

    project = re.sub(
        r"\s+",
        " ",
        project
    ).strip()

    return project


def remove_project_identity(row):

    text = str(
        row["log_text"]
    ).lower()

    project = clean_project_name(
        row["project_name"]
    )

    prefix = "project " + project

    if text.startswith(prefix):

        text = text[
            len(prefix):
        ].strip()

    return text


df["model_text"] = df.apply(
    remove_project_identity,
    axis=1
)

print(
    "\nProject identity removed "
    "from model input."
)


# ============================================================
# CREATE PARTIAL LOG REPRESENTATIONS
# ============================================================

def get_partial_text(
    text,
    percentage
):

    words = str(text).split()

    if len(words) == 0:
        return ""

    keep = max(
        1,
        int(
            len(words)
            * percentage
            / 100
        )
    )

    return " ".join(
        words[:keep]
    )


for percentage in [
    25,
    50,
    75,
    100
]:

    df[
        f"log_{percentage}"
    ] = df[
        "model_text"
    ].apply(
        lambda x:
        get_partial_text(
            x,
            percentage
        )
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# 64% train
# 16% validation
# 20% test
#
# The exact same split is reused for every model.
# ============================================================

train_val_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_val_df["label"]
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(val_df)
)

print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# SAVE SPLIT INFORMATION
#
# This is important because later DistilBERT and LoRA should
# use the same train/validation/test records.
# ============================================================

train_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "train_split.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "validation_split.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_split.csv"
    ),
    index=False
)


# ============================================================
# MODEL SIZE FUNCTION
# ============================================================

def get_model_size_mb(model):

    with tempfile.NamedTemporaryFile(
        suffix=".joblib",
        delete=False
    ) as temp_file:

        temp_path = temp_file.name

    try:

        joblib.dump(
            model,
            temp_path
        )

        size_mb = (
            os.path.getsize(
                temp_path
            )
            / (1024 * 1024)
        )

    finally:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )

    return size_mb


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

    elif hasattr(
        model,
        "decision_function"
    ):

        raw_score = (
            model
            .decision_function(X)
        )

        # Convert decision score to 0-1 scale
        return (
            1 /
            (
                1
                + np.exp(
                    -raw_score
                )
            )
        )

    else:

        return (
            model.predict(X)
        )


# ============================================================
# VALIDATION-BASED THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


# ============================================================
# DEFINE MODELS
# ============================================================

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


# ============================================================
# OPTIONAL STRONGER BASELINE
# ============================================================

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


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []

for percentage in [
    25,
    50,
    75,
    100
]:

    text_column = (
        f"log_{percentage}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"LOG AVAILABILITY: "
        f"{percentage}%"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        max_features=20000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True
    )

    X_train = (
        vectorizer
        .fit_transform(
            train_df[
                text_column
            ]
        )
    )

    X_val = (
        vectorizer
        .transform(
            val_df[
                text_column
            ]
        )
    )

    X_test = (
        vectorizer
        .transform(
            test_df[
                text_column
            ]
        )
    )

    y_train = (
        train_df[
            "label"
        ].values
    )

    y_val = (
        val_df[
            "label"
        ].values
    )

    y_test = (
        test_df[
            "label"
        ].values
    )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    for model_name, model in models.items():

        print(
            f"\nTraining "
            f"{model_name}..."
        )

        # ------------------------------------
        # Training time
        # ------------------------------------

        start_train = (
            time.perf_counter()
        )

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter()
            - start_train
        )


        # ------------------------------------
        # Validation probabilities
        # ------------------------------------

        val_scores = get_scores(
            model,
            X_val
        )

        best_threshold, validation_f1 = (
            find_best_threshold(
                y_val,
                val_scores
            )
        )


        # ------------------------------------
        # Test inference time
        # ------------------------------------

        start_prediction = (
            time.perf_counter()
        )

        test_scores = get_scores(
            model,
            X_test
        )

        prediction_time = (
            time.perf_counter()
            - start_prediction
        )

        prediction_time_per_build_ms = (

            prediction_time
            / len(y_test)
            * 1000
        )


        # ------------------------------------
        # Test prediction
        # ------------------------------------

        y_pred = (
            test_scores
            >= best_threshold
        ).astype(int)


        # ------------------------------------
        # Metrics
        # ------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        try:

            roc_auc = roc_auc_score(
                y_test,
                test_scores
            )

        except Exception:

            roc_auc = np.nan


        try:

            pr_auc = (
                average_precision_score(
                    y_test,
                    test_scores
                )
            )

        except Exception:

            pr_auc = np.nan


        cm = confusion_matrix(
            y_test,
            y_pred
        )


        # ------------------------------------
        # Model size
        # ------------------------------------

        model_size = (
            get_model_size_mb(
                model
            )
        )


        # ------------------------------------
        # Store result
        # ------------------------------------

        result = {

            "log_availability":
                f"{percentage}%",

            "model":
                model_name,

            "threshold":
                round(
                    best_threshold,
                    4
                ),

            "validation_f1":
                round(
                    validation_f1,
                    4
                ),

            "accuracy":
                round(
                    accuracy,
                    4
                ),

            "precision":
                round(
                    precision,
                    4
                ),

            "recall":
                round(
                    recall,
                    4
                ),

            "f1_score":
                round(
                    f1,
                    4
                ),

            "roc_auc":
                round(
                    roc_auc,
                    4
                )
                if not np.isnan(
                    roc_auc
                )
                else "",

            "pr_auc":
                round(
                    pr_auc,
                    4
                )
                if not np.isnan(
                    pr_auc
                )
                else "",

            "training_time_sec":
                round(
                    training_time,
                    4
                ),

            "prediction_time_per_build_ms":
                round(
                    prediction_time_per_build_ms,
                    6
                ),

            "model_size_mb":
                round(
                    model_size,
                    4
                ),

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

        all_results.append(
            result
        )

        print(
            "Threshold:",
            round(
                best_threshold,
                2
            )
        )

        print(
            "F1:",
            round(
                f1,
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


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_file = os.path.join(
    OUTPUT_FOLDER,
    "01_baseline_all_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# FULL-LOG RESULT ONLY
# ============================================================

full_log_results = (

    results_df[
        results_df[
            "log_availability"
        ]
        == "100%"
    ]
    .copy()
)

full_log_results = (
    full_log_results
    .sort_values(
        by="f1_score",
        ascending=False
    )
)

full_log_results.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "01_full_log_model_comparison.csv"
    ),

    index=False
)


# ============================================================
# SAVE DATASET / EXPERIMENT SUMMARY
# ============================================================

summary = {

    "total_records":
        int(
            len(df)
        ),

    "unique_projects":
        int(
            df[
                "project_name"
            ]
            .nunique()
        ),

    "training_records":
        int(
            len(train_df)
        ),

    "validation_records":
        int(
            len(val_df)
        ),

    "test_records":
        int(
            len(test_df)
        ),

    "successful_builds":
        int(
            (
                df["label"]
                == 0
            ).sum()
        ),

    "failed_builds":
        int(
            (
                df["label"]
                == 1
            ).sum()
        ),

    "failure_ratio":
        float(
            df[
                "label"
            ].mean()
        ),

    "project_identity_used_as_model_feature":
        False,

    "random_state":
        RANDOM_STATE
}


with open(

    os.path.join(
        OUTPUT_FOLDER,
        "01_experiment_summary.json"
    ),

    "w"

) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# DISPLAY FINAL FULL-LOG COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "100% LOG - FINAL BASELINE COMPARISON"
)

print(
    "=" * 70
)

columns_to_show = [

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

print(
    full_log_results[
        columns_to_show
    ].to_string(
        index=False
    )
)

print(
    "\nResults saved in:",
    OUTPUT_FOLDER
)

print(
    "\nMain result file:",
    results_file
)

print(
    "\nSTEP 4 COMPLETED SUCCESSFULLY"
)