from __future__ import annotations

import json
from pathlib import Path


EMBEDDINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "embeddings.json"


def load_embeddings() -> dict[str, list[float]]:
    if not EMBEDDINGS_PATH.exists():
        return {}

    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {str(label): [float(value) for value in embedding] for label, embedding in data.items()}


def save_embedding(label: str, embedding: list[float]) -> None:
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings()
    embeddings[label] = embedding

    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(embeddings, file)
