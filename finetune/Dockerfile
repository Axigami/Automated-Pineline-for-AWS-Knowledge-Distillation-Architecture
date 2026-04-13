# =============================================================
# Image: distillation-finetune
# Purpose: Fine-tune CNN teacher model on SageMaker
# Script: FineTuneTeacher.py
# =============================================================

FROM python:3.10-slim

# System deps for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    "numpy>=1.23.5,<2.0" \
    pandas>=1.5.0 \
    scikit-learn>=1.2.0 \
    boto3>=1.26.0

# SageMaker training toolkit (cho SM_CHANNEL_TRAINING, SM_MODEL_DIR, etc.)
RUN pip install --no-cache-dir sagemaker-training

# Copy script
COPY FineTuneTeacher.py /opt/ml/code/FineTuneTeacher.py

# SageMaker sẽ chạy file này
ENV SAGEMAKER_PROGRAM=FineTuneTeacher.py

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "/opt/ml/code/FineTuneTeacher.py"]
