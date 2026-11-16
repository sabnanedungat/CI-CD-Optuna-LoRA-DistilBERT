import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


# ============================================================
# FINAL STEADY-STATE KAFKA BENCHMARK
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_FILE = (
    BASE_DIR
    / "results_baseline"
    / "test_split.csv"
)

BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "cicd_logs_steady"

NUMBER_OF_BUILDS = 50

# 10 messages per second
MESSAGE_INTERVAL_SECONDS = 0.10


print("=" * 85)
print("STEADY-STATE KAFKA PRODUCER")
print("=" * 85)


df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)


stages = [
    (25, "log_25"),
    (50, "log_50"),
    (75, "log_75"),
    (100, "log_100")
]


producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVER,
    value_serializer=lambda value:
        json.dumps(value).encode("utf-8")
)


number_to_send = min(
    NUMBER_OF_BUILDS,
    len(df)
)

total_messages = (
    number_to_send
    * 4
)


print(
    "\nBuilds:",
    number_to_send
)

print(
    "Messages:",
    total_messages
)

print(
    "Target rate: 10 messages/sec"
)

print(
    "Topic:",
    TOPIC
)


start_time = time.perf_counter()

message_count = 0


for row_index in range(
    number_to_send
):

    row = df.iloc[
        row_index
    ]

    build_id = (
        f"steady_build_"
        f"{row_index + 1:03d}"
    )

    actual_label = int(
        row["label"]
    )


    for percentage, column in stages:

        message = {
            "build_id":
                build_id,

            "log_percentage":
                percentage,

            "log_text":
                str(
                    row[column]
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

        producer.flush()

        message_count += 1


        if (
            message_count % 20
            == 0
        ):

            print(
                f"Sent "
                f"{message_count}/"
                f"{total_messages}"
            )


        time.sleep(
            MESSAGE_INTERVAL_SECONDS
        )


producer.flush()
producer.close()


elapsed = (
    time.perf_counter()
    - start_time
)


print(
    "\n"
    + "=" * 85
)

print(
    "STEADY-STATE PRODUCER SUMMARY"
)

print(
    "=" * 85
)

print(
    "Messages sent:",
    message_count
)

print(
    "Duration:",
    round(
        elapsed,
        3
    ),
    "seconds"
)

print(
    "Actual send rate:",
    round(
        message_count
        / elapsed,
        3
    ),
    "messages/sec"
)

print(
    "\nSTEADY-STATE PRODUCER COMPLETED"
)