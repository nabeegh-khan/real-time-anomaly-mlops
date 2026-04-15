# Real-Time Anomaly Detection MLOps Pipeline

End-to-end streaming ML pipeline demonstrating production MLOps infrastructure:
Kafka ingestion → Spark feature engineering → LSTM autoencoder training →
FastAPI model serving → Airflow orchestration → Evidently AI drift monitoring.

---

## Architecture

```
NAB Dataset (58 time-series files)
        ↓
Apache Kafka — NAB replay producer · Confluent Cloud
        ↓
Spark Structured Streaming — sliding windows · rolling features
        ↓
DuckDB + dbt — feature store · batch transforms · schema tests
        ↓
┌─────────────────────┐     ┌─────────────────────┐
│ PyTorch LSTM        │     │ FastAPI + Docker     │
│ Autoencoder         │     │ /predict endpoint    │
│ MLflow tracking     │     │ anomaly flag + score │
└─────────────────────┘     └─────────────────────┘
        ↓                           ↓
        └──────── Apache Airflow ───┘
                 Training DAG · Drift DAG
                        ↓
                 Evidently AI
                 Drift monitoring · alerts
```

## Dataset

**Numenta Anomaly Benchmark (NAB)** — 58 labeled real-world time-series files
across IoT sensors, AWS CloudWatch metrics, Twitter volume, and traffic data.

- 38 series selected (5-minute sampling intervals, consistent resolution)
- 270,723 sliding windows (window size = 12 steps = 1 hour)
- 9.3% anomaly rate across all windows
- Train / Val / Test split: 70% / 15% / 15% (chronological per series)

---

## Model

**LSTM Autoencoder** — trained on normal segments only. Anomaly detection via
reconstruction error: windows the model cannot reconstruct well are flagged as anomalous.

| Component | Detail |
|-----------|--------|
| Architecture | Encoder LSTM (2 layers, hidden 64) → Decoder LSTM → Linear |
| Parameters | 118,983 trainable |
| Training | 20 epochs, Adam lr=1e-3, MSE loss, normal windows only |
| Best val loss | 0.000081 |
| ROC-AUC (test) | 0.6424 |
| Threshold | 95th percentile of normal reconstruction error (optimized on val) |

All experiments tracked in MLflow with hyperparameters, loss curves, and model artifacts.

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.6424 |
| F1 (optimized threshold) | 0.1636 |
| Precision | 0.1070 |
| Recall | 0.3471 |
| Dataset drift (train vs test) | Not detected (0/7 features) |

ROC-AUC of 0.64 confirms genuine discriminative ability above random (0.50).
Low F1 reflects the inherent difficulty of NAB: many anomaly windows are subtle
regime changes that overlap with normal variance — a known challenge in unsupervised
time-series anomaly detection.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Streaming ingestion | Apache Kafka (Confluent Cloud) |
| Stream processing | Apache Spark Structured Streaming 3.5 |
| Feature store | DuckDB + dbt |
| Model training | PyTorch 2.5 · LSTM Autoencoder |
| Experiment tracking | MLflow 3.11 |
| Model serving | FastAPI + Docker |
| Orchestration | Apache Airflow |
| Drift monitoring | Evidently AI 0.7 |
| Language | Python 3.10 |

---

## Repo Structure

```
├── data/
│   ├── raw/NAB/                  # Numenta Anomaly Benchmark (clone separately)
│   └── processed/                # feature store, model artifacts, reports
├── infrastructure/
│   ├── kafka/docker-compose.yml  # Kafka + Zookeeper
│   └── airflow/docker-compose.yml
├── src/
│   ├── ingestion/                # Kafka NAB replay producer
│   ├── streaming/                # Spark Structured Streaming job
│   ├── transforms/               # dbt project (DuckDB)
│   ├── training/                 # LSTM Autoencoder + MLflow
│   ├── serving/                  # FastAPI + Dockerfile
│   ├── monitoring/               # Evidently drift reports
│   └── orchestration/dags/       # Airflow DAGs
├── notebooks/
│   ├── 01_nab_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lstm_autoencoder.ipynb
│   └── 04_evaluation_monitoring.ipynb
└── environment.yml
```

## Setup

**Prerequisites:** Anaconda, Docker Desktop, Java 11

```bash
# 1. Clone the repo
git clone https://github.com/nabeegh-khan/real-time-anomaly-mlops.git
cd real-time-anomaly-mlops

# 2. Clone NAB dataset
git clone https://github.com/numenta/NAB.git data/raw/NAB

# 3. Create conda environment
conda env create -f environment.yml
conda activate mlops_pipeline

# 4. Start Kafka
cd infrastructure/kafka && docker-compose up -d && cd ../..

# 5. Run notebooks in order (01 → 02 → 03)

# 6. Start FastAPI server
cd src/serving && uvicorn main:app --port 8000

# 7. Start Spark streaming job
python src/streaming/spark_job.py

# 8. Start Kafka producer
python src/ingestion/nab_producer.py

# 9. Generate drift report
python src/monitoring/drift_report.py
```

---

## AI Assistance Disclosure

This project was developed with assistance from Claude (Anthropic) for code
scaffolding, debugging, and documentation. All architectural decisions, dataset
selection rationale, model design choices, and result interpretation are my own.

---

*Personal portfolio project — not affiliated with the University of Toronto.*