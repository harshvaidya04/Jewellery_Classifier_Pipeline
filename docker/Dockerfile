FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV PYTHONUNBUFFERED=1 \
    SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python", "train.py"]
