from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.ml.model import image_to_embedding, load_model
from app.ml.similarity import cosine_similarity
from app.storage.embeddings import load_embeddings, save_embedding
from app.storage.metadata import load_metadata, save_metadata


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "images"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {"jpg", "jpeg", "png", "webp"}
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

app = FastAPI(title="Petroglyph ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_model() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    load_model()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/register/{label}")
async def register_image(
    label: str,
    file: Annotated[UploadFile, File(description="jpg, png, or webp image")],
    title: Annotated[str, Form(max_length=100)],
    description: Annotated[str, Form(max_length=250)],
) -> dict[str, bool | str]:
    safe_label = validate_label(label)
    image, suffix = await read_image_upload(file)

    image_path = IMAGE_DIR / f"{safe_label}.{suffix}"
    image.save(image_path)

    embedding = image_to_embedding(image)
    save_embedding(safe_label, embedding)
    save_metadata(safe_label, title, description)

    return {
        "success": True,
        "label": safe_label,
        "title": title,
        "description": description,
    }


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="jpg, png, or webp image")],
) -> dict[str, str | float | dict[str, float] | None]:
    registered = load_embeddings()
    if not registered:
        raise HTTPException(status_code=400, detail="No reference embeddings registered")

    image, _ = await read_image_upload(file)
    scan_embedding = image_to_embedding(image)

    scores = {
        label: round(cosine_similarity(scan_embedding, embedding), 4)
        for label, embedding in registered.items()
    }
    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]
    prediction = best_label if confidence >= SIMILARITY_THRESHOLD else "unknown"
    metadata = load_metadata().get(prediction) if prediction != "unknown" else None

    return {
        "prediction": prediction,
        "confidence": confidence,
        "title": metadata["title"] if metadata else None,
        "description": metadata["description"] if metadata else None,
        "scores": scores,
    }


def validate_label(label: str) -> str:
    if not LABEL_PATTERN.fullmatch(label):
        raise HTTPException(
            status_code=400,
            detail="Label must be 1-64 characters: letters, numbers, underscore, or hyphen",
        )
    return label


async def read_image_upload(file: UploadFile) -> tuple[Image.Image, str]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only jpg, png, and webp images are supported")

    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix == "jpeg":
        suffix = "jpg"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Image filename must end in jpg, png, or webp")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as img:
            img.load()
            image = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from None

    return image, suffix
