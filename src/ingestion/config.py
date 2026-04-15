# Kafka connection settings and topic configuration
# kept separate so producer and consumer both import from one place

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# one topic per data stream
TOPIC_NAB_RAW = "nab-raw-stream"

# how fast to replay NAB data
# 1.0 = real time (one message per 5 minutes)
# 0.0 = as fast as possible (for testing)
REPLAY_SPEED_SECONDS = 0.1  # 100ms between messages during dev

# which NAB categories to stream
STREAM_CATEGORIES = [
    "realAWSCloudwatch",
    "artificialWithAnomaly",
]