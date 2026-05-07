from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


@lru_cache(maxsize=1)
def load_model() -> tuple[torch.nn.Module, Any, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model.to(device)
    model.eval()
    return model, preprocess, device


def image_to_embedding(image: Image.Image) -> list[float]:
    model, preprocess, device = load_model()

    # OpenCLIP maps the image into a shared semantic vector space. L2-normalizing
    # here lets cosine similarity become a simple, stable vector comparison.
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    embedding = features.squeeze(0).cpu().numpy().astype(np.float32)
    return embedding.tolist()
