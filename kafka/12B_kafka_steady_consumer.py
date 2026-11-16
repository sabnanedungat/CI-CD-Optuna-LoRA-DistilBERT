import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kafka import KafkaConsumer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel


# ============================================================
# STEP 16A
# KAFKA PERFORMANCE BENCHMARK CONSUMER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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
    / "results_kafka_performance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BOOTSTRAP_SERVER = "localhost:9092"
INPUT_TOPIC = "cicd_logs_steady"

MAX_LENGTH = 128

EXPECTED_MESSAGES = 200


THRESHOLDS = {
    25: 0.42,
    50: 0.60,
    75: 0.79,
    100: 0.59
}


print("=" * 90)
print("STEP 16A - KAFKA PERFORMANCE BENCHMARK CONSUMER")
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
# LOAD MODEL
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
    "Loading DistilBERT..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)


print(
    "Loading Optuna-LoRA..."
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
# WARM-UP MODEL
# ============================================================

print(
    "Warming up model..."
)

warmup_text = (
    "lang ruby branch master "
    "duration 100 tests_ran 50"
)

for _ in range(10):

    encoded = tokenizer(
        warmup_text,
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

        _ = model(
            **encoded
        )


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    INPUT_TOPIC,

    bootstrap_servers=BOOTSTRAP_SERVER,

    auto_offset_reset="latest",

    enable_auto_commit=False,

    group_id="jss-steady-benchmark-v1",

    value_deserializer=lambda value:
        json.loads(
            value.decode(
                "utf-8"
            )
        )
)


print(
    "\nKafka server:",
    BOOTSTRAP_SERVER
)

print(
    "Benchmark topic:",
    INPUT_TOPIC
)

print(
    "Expected messages:",
    EXPECTED_MESSAGES
)

print(
    "\nWaiting for benchmark producer..."
)


# ============================================================
# RESULT STORAGE
# ============================================================

records = []

benchmark_start = None
benchmark_end = None


# ============================================================
# PROCESS BENCHMARK MESSAGES
# ============================================================

for message in consumer:

    data = message.value

    if benchmark_start is None:

        benchmark_start = (
            time.perf_counter()
        )


    received_timestamp = (
        time.time()
    )


    build_id = data.get(
        "build_id",
        "unknown"
    )

    log_percentage = int(
        data.get(
            "log_percentage",
            100
        )
    )

    log_text = str(
        data.get(
            "log_text",
            ""
        )
    )

    actual_label = int(
        data.get(
            "actual_label",
            -1
        )
    )

    sent_timestamp = float(
        data.get(
            "sent_timestamp",
            received_timestamp
        )
    )


    # ========================================================
    # TOKENIZATION
    # ========================================================

    preprocessing_start = (
        time.perf_counter()
    )

    encoded = tokenizer(
        log_text,
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

    preprocessing_ms = (
        (
            time.perf_counter()
            - preprocessing_start
        )
        * 1000
    )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    inference_start = (
        time.perf_counter()
    )

    with torch.no_grad():

        outputs = model(
            **encoded
        )

    if device.type == "cuda":

        torch.cuda.synchronize()


    inference_ms = (
        (
            time.perf_counter()
            - inference_start
        )
        * 1000
    )


    probability = float(
        torch.softmax(
            outputs.logits,
            dim=1
        )[0, 1]
        .detach()
        .cpu()
        .item()
    )


    threshold = THRESHOLDS.get(
        log_percentage,
        0.59
    )


    prediction = int(
        probability
        >= threshold
    )


    processing_finished_timestamp = (
        time.time()
    )


    # ========================================================
    # LATENCY
    # ========================================================

    end_to_end_ms = (
        processing_finished_timestamp
        - sent_timestamp
    ) * 1000


    total_local_processing_ms = (
        preprocessing_ms
        + inference_ms
    )


    correct = (

        prediction
        == actual_label

        if actual_label
        in [0, 1]

        else None
    )


    records.append(
        {
            "build_id":
                build_id,

            "log_percentage":
                log_percentage,

            "actual_label":
                actual_label,

            "failure_probability":
                probability,

            "threshold":
                threshold,

            "prediction":
                prediction,

            "correct":
                correct,

            "tokenization_ms":
                preprocessing_ms,

            "inference_ms":
                inference_ms,

            "local_processing_ms":
                total_local_processing_ms,

            "end_to_end_ms":
                end_to_end_ms,

            "sent_timestamp":
                sent_timestamp,

            "received_timestamp":
                received_timestamp,

            "completed_timestamp":
                processing_finished_timestamp
        }
    )


    count = len(
        records
    )


    if (
        count % 20
        == 0
    ):

        print(
            f"Processed "
            f"{count}/"
            f"{EXPECTED_MESSAGES}"
        )


    if (
        count
        >= EXPECTED_MESSAGES
    ):

        benchmark_end = (
            time.perf_counter()
        )

        break


# ============================================================
# CLOSE CONSUMER
# ============================================================

consumer.close()


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    records
)

raw_file = (
    OUTPUT_DIR
    / "12_kafka_message_level_results.csv"
)

results_df.to_csv(
    raw_file,
    index=False
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

elapsed_seconds = (
    benchmark_end
    - benchmark_start
)


throughput_messages_sec = (
    len(results_df)
    / elapsed_seconds
)


unique_builds = (
    results_df[
        "build_id"
    ]
    .nunique()
)


throughput_builds_sec = (
    unique_builds
    / elapsed_seconds
)


def latency_summary(
    series
):

    return {

        "mean":
            float(
                series.mean()
            ),

        "median":
            float(
                series.median()
            ),

        "p95":
            float(
                np.percentile(
                    series,
                    95
                )
            ),

        "p99":
            float(
                np.percentile(
                    series,
                    99
                )
            ),

        "min":
            float(
                series.min()
            ),

        "max":
            float(
                series.max()
            )
    }


inference_stats = latency_summary(
    results_df[
        "inference_ms"
    ]
)


processing_stats = latency_summary(
    results_df[
        "local_processing_ms"
    ]
)


e2e_stats = latency_summary(
    results_df[
        "end_to_end_ms"
    ]
)


summary = {

    "messages_processed":
        len(
            results_df
        ),

    "unique_builds":
        unique_builds,

    "benchmark_duration_sec":
        elapsed_seconds,

    "throughput_messages_per_sec":
        throughput_messages_sec,

    "throughput_builds_per_sec":
        throughput_builds_sec,

    "inference_mean_ms":
        inference_stats[
            "mean"
        ],

    "inference_median_ms":
        inference_stats[
            "median"
        ],

    "inference_p95_ms":
        inference_stats[
            "p95"
        ],

    "inference_p99_ms":
        inference_stats[
            "p99"
        ],

    "processing_mean_ms":
        processing_stats[
            "mean"
        ],

    "processing_p95_ms":
        processing_stats[
            "p95"
        ],

    "e2e_mean_ms":
        e2e_stats[
            "mean"
        ],

    "e2e_median_ms":
        e2e_stats[
            "median"
        ],

    "e2e_p95_ms":
        e2e_stats[
            "p95"
        ],

    "e2e_p99_ms":
        e2e_stats[
            "p99"
        ]
}


summary_df = pd.DataFrame(
    [
        summary
    ]
)


summary_file = (
    OUTPUT_DIR
    / "12_kafka_performance_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PERFORMANCE BY LOG STAGE
# ============================================================

stage_summary = (

    results_df
    .groupby(
        "log_percentage"
    )
    .agg(

        messages=(
            "build_id",
            "size"
        ),

        mean_inference_ms=(
            "inference_ms",
            "mean"
        ),

        median_inference_ms=(
            "inference_ms",
            "median"
        ),

        mean_e2e_ms=(
            "end_to_end_ms",
            "mean"
        ),

        accuracy=(
            "correct",
            "mean"
        )
    )
    .reset_index()
)


stage_summary.to_csv(
    OUTPUT_DIR
    / "12_kafka_stage_summary.csv",
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "KAFKA PERFORMANCE SUMMARY"
)

print(
    "=" * 90
)


print(
    "Messages processed:",
    len(
        results_df
    )
)

print(
    "Unique builds:",
    unique_builds
)

print(
    "Benchmark duration:",
    round(
        elapsed_seconds,
        3
    ),
    "seconds"
)

print(
    "\nThroughput:"
)

print(
    "Messages/sec:",
    round(
        throughput_messages_sec,
        3
    )
)

print(
    "Builds/sec:",
    round(
        throughput_builds_sec,
        3
    )
)


print(
    "\nModel inference latency:"
)

print(
    "Mean:",
    round(
        inference_stats[
            "mean"
        ],
        3
    ),
    "ms"
)

print(
    "Median:",
    round(
        inference_stats[
            "median"
        ],
        3
    ),
    "ms"
)

print(
    "P95:",
    round(
        inference_stats[
            "p95"
        ],
        3
    ),
    "ms"
)

print(
    "P99:",
    round(
        inference_stats[
            "p99"
        ],
        3
    ),
    "ms"
)


print(
    "\nEnd-to-end Kafka latency:"
)

print(
    "Mean:",
    round(
        e2e_stats[
            "mean"
        ],
        3
    ),
    "ms"
)

print(
    "Median:",
    round(
        e2e_stats[
            "median"
        ],
        3
    ),
    "ms"
)

print(
    "P95:",
    round(
        e2e_stats[
            "p95"
        ],
        3
    ),
    "ms"
)

print(
    "P99:",
    round(
        e2e_stats[
            "p99"
        ],
        3
    ),
    "ms"
)


print(
    "\n"
    + "=" * 90
)

print(
    "PERFORMANCE BY LOG AVAILABILITY"
)

print(
    "=" * 90
)

print(
    stage_summary
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nResults saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSTEP 16A COMPLETED SUCCESSFULLY"
)
