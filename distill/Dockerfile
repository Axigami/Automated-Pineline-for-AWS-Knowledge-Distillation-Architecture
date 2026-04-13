# =============================================================
# Image: distillation-student
# Purpose: Train LightGBM student via knowledge distillation
# Script: IOT-PROJECT.py
# =============================================================

FROM python:3.10-slim

# System deps for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — nhẹ hơn nhiều vì không cần TensorFlow
RUN pip install --no-cache-dir \
    lightgbm>=4.0.0 \
    "numpy>=1.23.5,<2.0" \
    pandas>=1.5.0 \
    scikit-learn>=1.2.0 \
    joblib>=1.2.0 \
    boto3>=1.26.0

# SageMaker training toolkit
RUN pip install --no-cache-dir sagemaker-training

# Copy script
COPY IOT-PROJECT.py /opt/ml/code/IOT-PROJECT.py

ENV SAGEMAKER_PROGRAM=IOT-PROJECT.py

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "/opt/ml/code/IOT-PROJECT.py"]
