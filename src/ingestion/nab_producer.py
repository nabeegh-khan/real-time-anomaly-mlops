# reads NAB CSV files and publishes rows to Kafka one at a time
# simulates a live IoT sensor stream by replaying historical data
import sys
import time
import json
from pathlib import Path

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

# add project root to path so we can import config
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_NAB_RAW,
    REPLAY_SPEED_SECONDS,
    STREAM_CATEGORIES,
)

import pandas as pd

NAB_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "NAB" / "data"


def create_topic_if_missing(topic_name: str):
    """Create the Kafka topic if it doesn't already exist."""
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    existing = admin.list_topics(timeout=5).topics

    if topic_name not in existing:
        topic = NewTopic(topic_name, num_partitions=1, replication_factor=1)
        fs = admin.create_topics([topic])
        for t, f in fs.items():
            try:
                f.result()
                print(f"Created topic: {t}")
            except Exception as e:
                print(f"Topic creation error: {e}")
    else:
        print(f"Topic already exists: {topic_name}")


def delivery_report(err, msg):
    """Called once per message to confirm delivery or log errors."""
    if err:
        print(f"Delivery failed: {err}")


def load_nab_series(categories: list[str]) -> pd.DataFrame:
    """Load all CSV files for the specified categories into one dataframe."""
    frames = []
    for category in categories:
        category_path = NAB_DATA_PATH / category
        for csv_file in sorted(category_path.glob("*.csv")):
            df = pd.read_csv(csv_file, parse_dates=["timestamp"])
            df["category"]    = category
            df["series_name"] = csv_file.stem
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # sort by timestamp so we stream in chronological order
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def run_producer():
    """Main producer loop — reads NAB rows and publishes to Kafka."""
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    create_topic_if_missing(TOPIC_NAB_RAW)

    print(f"Loading NAB data for categories: {STREAM_CATEGORIES}")
    df = load_nab_series(STREAM_CATEGORIES)
    print(f"Loaded {len(df):,} rows — starting stream...")

    for idx, row in df.iterrows():
        # build the message payload
        message = {
            "timestamp":   row["timestamp"].isoformat(),
            "value":       float(row["value"]),
            "category":    row["category"],
            "series_name": row["series_name"],
        }

        # publish to Kafka — key by series name so same series
        # always goes to the same partition
        producer.produce(
            topic=TOPIC_NAB_RAW,
            key=row["series_name"],
            value=json.dumps(message),
            callback=delivery_report,
        )

        # poll to trigger delivery callbacks without blocking
        producer.poll(0)

        # print progress every 500 messages
        if idx % 500 == 0:
            print(f"  Published {idx:,} / {len(df):,} messages")

        time.sleep(REPLAY_SPEED_SECONDS)

    # flush any remaining messages before exiting
    producer.flush()
    print("Producer finished — all messages sent")


if __name__ == "__main__":
    run_producer()