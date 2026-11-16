import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


# ============================================================
# STEP 16B
# KAFKA PERFORMANCE BENCHMARK PRODUCER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_FILE = (
    BASE_DIR
    / "results_baseline"
    / "test_split.csv"
)

BOOTSTRAP_SERVER = "localhost:9092"

TOPIC = "cicd_logs_benchmark"

NUMBER_OF_BUILDS = 50


print("=" * 90)
print("STEP 16B - KAFKA PERFORMANCE BENCHMARK PRODUCER")
print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )


df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)


required_columns = [
    "log_25",
    "log_50",
    "log_75",
    "log_100",
    "label"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# ============================================================
# PRODUCER
# ============================================================

producer = KafkaProducer(

    bootstrap_servers=BOOTSTRAP_SERVER,

    value_serializer=lambda value:
        json.dumps(
            value
        ).encode(
            "utf-8"
        )
)


stages = [

    (25, "log_25"),

    (50, "log_50"),

    (75, "log_75"),

    (100, "log_100")
]


number_of_builds = min(
    NUMBER_OF_BUILDS,
    len(df)
)


total_messages = (
    number_of_builds
    * len(stages)
)


print(
    "\nBuilds:",
    number_of_builds
)

print(
    "Messages:",
    total_messages
)

print(
    "Topic:",
    TOPIC
)

print(
    "\nStarting benchmark..."
)


# ============================================================
# SEND MESSAGES RAPIDLY
# ============================================================

benchmark_start = (
    time.perf_counter()
)


message_count = 0


for row_index in range(
    number_of_builds
):

    row = df.iloc[
        row_index
    ]

    build_id = (
        f"benchmark_build_"
        f"{row_index + 1:03d}"
    )

    actual_label = int(
        row[
            "label"
        ]
    )


    for percentage, column in stages:

        message = {

            "build_id":
                build_id,

            "log_percentage":
                percentage,

            "log_text":
                str(
                    row[
                        column
                    ]
                ),

            "actual_label":
                actual_label,

            "sent_timestamp":
                time.time()
        }


        producer.send(
            TOPIC,
            message
        )


        message_count += 1


        if (
            message_count
            % 20
            == 0
        ):

            print(
                f"Sent "
                f"{message_count}/"
                f"{total_messages}"
            )


# ============================================================
# ENSURE DELIVERY
# ============================================================

producer.flush()


elapsed = (
    time.perf_counter()
    - benchmark_start
)


producer.close()


# ============================================================
# SUMMARY
# ============================================================

producer_rate = (
    message_count
    / elapsed
)


print(
    "\n"
    + "=" * 90
)

print(
    "BENCHMARK PRODUCER SUMMARY"
)

print(
    "=" * 90
)

print(
    "Messages sent:",
    message_count
)

print(
    "Builds sent:",
    number_of_builds
)

print(
    "Producer duration:",
    round(
        elapsed,
        3
    ),
    "seconds"
)

print(
    "Producer send rate:",
    round(
        producer_rate,
        3
    ),
    "messages/sec"
)

print(
    "\nSTEP 16B COMPLETED SUCCESSFULLY"
)