from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


METADATA_PATH = Path(__file__).resolve().parents[1] / "data" / "metadata.json"


class LabelMetadata(TypedDict):
    title: str
    description: str


def load_metadata() -> dict[str, LabelMetadata]:
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    metadata: dict[str, LabelMetadata] = {}
    for label, item in data.items():
        if not isinstance(item, dict):
            continue
        metadata[str(label)] = {
            "title": str(item.get("title", "")),
            "description": str(item.get("description", "")),
        }
    return metadata


def save_metadata(label: str, title: str, description: str) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    metadata[label] = {"title": title, "description": description}

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False)
