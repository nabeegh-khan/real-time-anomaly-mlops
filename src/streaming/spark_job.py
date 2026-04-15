# Spark Structured Streaming job
# consumes from Kafka, computes rolling features, calls FastAPI /predict
import sys
import json
import requests
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, avg, stddev,
    lag, udf, pandas_udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, TimestampType, IntegerType
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]
from src.ingestion.config import KAFKA_BOOTSTRAP_SERVERS, TOPIC_NAB_RAW

# FastAPI endpoint
PREDICT_URL = "http://127.0.0.1:8000/predict"

# schema of the JSON messages coming from the producer
MESSAGE_SCHEMA = StructType([
    StructField("timestamp",   TimestampType(), True),
    StructField("value",       DoubleType(),    True),
    StructField("category",    StringType(),    True),
    StructField("series_name", StringType(),    True),
])


def create_spark_session():
    """Create a local Spark session with Kafka connector."""
    return (
        SparkSession.builder
        .appName("NAB Anomaly Detection Stream")
        .master("local[2]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
        )
        # reduce Spark logging noise during dev
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def run_streaming_job():
    print("Starting Spark Structured Streaming job...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # read from Kafka topic
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC_NAB_RAW)
        .option("startingOffsets", "latest")
        .load()
    )

    # deserialize JSON payload
    parsed = (
        raw_stream
        .select(
            from_json(
                col("value").cast("string"),
                MESSAGE_SCHEMA
            ).alias("data")
        )
        .select("data.*")
    )

    # compute rolling features using a 1-hour tumbling window (12 x 5-min steps)
    windowed = (
        parsed
        .withWatermark("timestamp", "10 minutes")
        .groupBy(
            col("series_name"),
            col("category"),
            window(col("timestamp"), "60 minutes", "5 minutes")
        )
        .agg(
            avg("value").alias("rolling_mean"),
            stddev("value").alias("rolling_std"),
        )
    )

    # write windowed aggregations to console so we can see them during dev
    query = (
        windowed
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", False)
        .option("numRows", 10)
        .trigger(processingTime="30 seconds")
        .start()
    )

    print("Streaming job running — waiting for data from Kafka...")
    print("Press Ctrl+C to stop")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_job()