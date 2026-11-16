import json
import time
from pathlib import Path
from datetime import datetime

import torch

from kafka import KafkaConsumer, KafkaProducer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel


# ============================================================
# STEP 15
# KAFKA OPTUNA-LoRA DISTILBERT CONSUMER
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

INPUT_TOPIC = "cicd_logs"

OUTPUT_TOPIC = "cicd_predictions"

BOOTSTRAP_SERVER = "localhost:9092"

MAX_LENGTH = 128


# ============================================================
# VALIDATION-SELECTED PARTIAL LOG THRESHOLDS
# ============================================================

THRESHOLDS = {
    25: 0.42,
    50: 0.60,
    75: 0.79,
    100: 0.59
}


print("=" * 85)
print("STEP 15 - KAFKA OPTUNA-LoRA CONSUMER")
print("=" * 85)


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
# CHECK MODEL FILES
# ============================================================

if not BASE_MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Base DistilBERT model missing:\n"
        f"{BASE_MODEL_DIR}"
    )


if not LORA_MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Optuna-LoRA adapter missing:\n"
        f"{LORA_MODEL_DIR}"
    )


# ============================================================
# LOAD TOKENIZER
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


# ============================================================
# LOAD DISTILBERT BASE
# ============================================================

print(
    "Loading fine-tuned DistilBERT..."
)

base_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL_DIR
    )
)


# ============================================================
# LOAD OPTUNA-LoRA
# ============================================================

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
# PREDICTION FUNCTION
# ============================================================

def predict_failure(
    log_text,
    log_percentage
):

    encoded = tokenizer(
        str(log_text),
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


    start_time = (
        time.perf_counter()
    )


    with torch.no_grad():

        outputs = model(
            **encoded
        )


    probability = (
        torch.softmax(
            outputs.logits,
            dim=1
        )[0, 1]
        .detach()
        .cpu()
        .item()
    )


    inference_time_ms = (
        (
            time.perf_counter()
            - start_time
        )
        * 1000
    )


    threshold = THRESHOLDS.get(
        int(log_percentage),
        THRESHOLDS[100]
    )


    predicted_label = int(
        probability
        >= threshold
    )


    predicted_state = (
        "FAILURE"
        if predicted_label == 1
        else "SUCCESS"
    )


    return {
        "failure_probability":
            float(
                probability
            ),

        "threshold":
            float(
                threshold
            ),

        "predicted_label":
            predicted_label,

        "predicted_state":
            predicted_state,

        "inference_time_ms":
            float(
                inference_time_ms
            )
    }


# ============================================================
# KAFKA PRODUCER FOR PREDICTIONS
# ============================================================

prediction_producer = KafkaProducer(

    bootstrap_servers=BOOTSTRAP_SERVER,

    value_serializer=lambda value:
        json.dumps(
            value
        ).encode(
            "utf-8"
        )
)


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    INPUT_TOPIC,

    bootstrap_servers=BOOTSTRAP_SERVER,

    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="optuna-lora-cicd-consumer",

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
    "Input topic:",
    INPUT_TOPIC
)

print(
    "Output topic:",
    OUTPUT_TOPIC
)

print(
    "\nWaiting for CI/CD log messages..."
)

print(
    "Press CTRL+C to stop."
)


# ============================================================
# PROCESS STREAM
# ============================================================

try:

    for message in consumer:

        data = (
            message.value
        )


        # ----------------------------------------------------
        # EXPECTED MESSAGE STRUCTURE
        #
        # {
        #   "build_id": "...",
        #   "log_percentage": 25,
        #   "log_text": "..."
        # }
        # ----------------------------------------------------


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


        log_text = data.get(
            "log_text",
            ""
        )


        if not log_text:

            print(
                "\nSkipped empty log:"
                f" {build_id}"
            )

            continue


        if log_percentage not in THRESHOLDS:

            print(
                f"\nUnsupported percentage "
                f"{log_percentage}%."
            )

            print(
                "Using 100% threshold."
            )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        result = predict_failure(
            log_text,
            log_percentage
        )


        # ----------------------------------------------------
        # EARLY WARNING
        # ----------------------------------------------------

        early_warning = bool(

            result[
                "predicted_label"
            ] == 1

            and

            log_percentage < 100
        )


        # ----------------------------------------------------
        # EARLY STOP SIGNAL
        #
        # This is only a recommendation signal.
        # It does NOT automatically terminate a CI build.
        # ----------------------------------------------------

        early_stop_signal = bool(
            early_warning
        )


        prediction_message = {

            "build_id":
                build_id,

            "log_percentage":
                log_percentage,

            "failure_probability":
                round(
                    result[
                        "failure_probability"
                    ],
                    6
                ),

            "threshold":
                result[
                    "threshold"
                ],

            "predicted_label":
                result[
                    "predicted_label"
                ],

            "predicted_state":
                result[
                    "predicted_state"
                ],

            "early_warning":
                early_warning,

            "early_stop_signal":
                early_stop_signal,

            "inference_time_ms":
                round(
                    result[
                        "inference_time_ms"
                    ],
                    3
                ),

            "timestamp":
                datetime.now()
                .isoformat()
        }


        # ----------------------------------------------------
        # SEND PREDICTION
        # ----------------------------------------------------

        prediction_producer.send(
            OUTPUT_TOPIC,
            prediction_message
        )

        prediction_producer.flush()


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 85
        )

        print(
            "Build ID:",
            build_id
        )

        print(
            "Log available:",
            f"{log_percentage}%"
        )

        print(
            "Failure probability:",
            round(
                result[
                    "failure_probability"
                ],
                4
            )
        )

        print(
            "Threshold:",
            result[
                "threshold"
            ]
        )

        print(
            "Prediction:",
            result[
                "predicted_state"
            ]
        )

        print(
            "Early warning:",
            early_warning
        )

        print(
            "Inference time:",
            round(
                result[
                    "inference_time_ms"
                ],
                3
            ),
            "ms"
        )

        print(
            "Prediction published to:",
            OUTPUT_TOPIC
        )


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

except KeyboardInterrupt:

    print(
        "\nStopping Kafka consumer..."
    )


finally:

    consumer.close()

    prediction_producer.flush()

    prediction_producer.close()


    print(
        "Kafka consumer closed successfully."
    )