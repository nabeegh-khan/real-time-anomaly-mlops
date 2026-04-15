FROM python:3.10-slim

WORKDIR /app

# copy requirements first so Docker caches the layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source files
COPY src/serving/main.py .
COPY src/serving/predict.py .
COPY src/training/model.py .

# copy model weights and config
COPY data/processed/model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]