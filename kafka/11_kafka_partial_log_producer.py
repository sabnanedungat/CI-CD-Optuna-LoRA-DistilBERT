import json
import time
from pathlib import Path

import pandas as pd

from kafka import KafkaProducer


# ============================================================
# STEP 15B
# PARTIAL-LOG KAFKA PRODUCER
#
# Sends the same CI build progressively at:
# 25%, 50%, 75%, 100%
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_FILE = (
    BASE_DIR
    / "results_baseline"
    / "test_split.csv"
)

BOOTSTRAP_SERVER = "localhost:9092"

TOPIC = "cicd_logs"

# Number of builds to demonstrate
NUMBER_OF_BUILDS = 5

# Delay between partial-log stages
DELAY_SECONDS = 1


print("=" * 85)
print("STEP 15B - KAFKA PARTIAL-LOG PRODUCER")
print("=" * 85)


# ============================================================
# CHECK DATA
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


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


print(
    "\nTest rows available:",
    len(df)
)


# ============================================================
# KAFKA PRODUCER
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


print(
    "\nKafka server:",
    BOOTSTRAP_SERVER
)

print(
    "Topic:",
    TOPIC
)


# ============================================================
# PARTIAL LOG MAPPING
# ============================================================

partial_levels = [

    (
        25,
        "log_25"
    ),

    (
        50,
        "log_50"
    ),

    (
        75,
        "log_75"
    ),

    (
        100,
        "log_100"
    )
]


# ============================================================
# SEND BUILDS
# ============================================================

number_to_send = min(
    NUMBER_OF_BUILDS,
    len(df)
)


print(
    "\nSending",
    number_to_send,
    "builds"
)

print(
    "Each build will be sent at "
    "25%, 50%, 75%, and 100% availability."
)


for row_index in range(
    number_to_send
):

    row = df.iloc[
        row_index
    ]

    build_id = (
        f"build_{row_index + 1:03d}"
    )

    actual_label = int(
        row[
            "label"
        ]
    )

    actual_state = (
        "FAILURE"
        if actual_label == 1
        else "SUCCESS"
    )


    print(
        "\n"
        + "=" * 85
    )

    print(
        "BUILD:",
        build_id
    )

    print(
        "Actual state:",
        actual_state
    )

    print(
        "=" * 85
    )


    for percentage, column in partial_levels:

        log_text = str(
            row[
                column
            ]
        )


        message = {

            "build_id":
                build_id,

            "log_percentage":
                percentage,

            "log_text":
                log_text,

            "actual_label":
                actual_label,

            "actual_state":
                actual_state,

            "sent_timestamp":
                time.time()
        }


        producer.send(
            TOPIC,
            message
        )

        producer.flush()


        print(
            f"Sent {build_id} "
            f"at {percentage}%"
        )


        time.sleep(
            DELAY_SECONDS
        )


producer.flush()

producer.close()


print(
    "\n"
    + "=" * 85
)

print(
    "PRODUCER COMPLETED"
)

print(
    "=" * 85
)

print(
    "All partial-log messages sent successfully."
)

print(
    "\nSTEP 15B COMPLETED SUCCESSFULLY"
)