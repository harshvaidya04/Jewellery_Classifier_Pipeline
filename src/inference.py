import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import build_model

# SageMaker calls model_fn, input_fn, predict_fn, output_fn

def model_fn(model_dir: str):
    model_dir = Path(model_dir)
    with open(model_dir / "classes.json") as f:
        classes = json.load(f)

    model = build_model("efficientnet_b3", num_classes=len(classes))
    state = torch.load(model_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"model": model.to(device), "classes": classes, "device": device}


def input_fn(request_body: bytes, content_type: str):
    arr = np.frombuffer(request_body, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def predict_fn(image: np.ndarray, model_assets: dict):
    model   = model_assets["model"]
    classes = model_assets["classes"]
    device  = model_assets["device"]

    transform = A.Compose([
        A.Resize(300, 300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    return {"classes": classes, "probabilities": probs}


def output_fn(prediction: dict, accept: str):
    return json.dumps(prediction), "application/json"
