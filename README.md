# Petroglyph ML Backend

Minimal FastAPI backend for registering petroglyph reference images and comparing scan uploads with OpenCLIP image embeddings.

## Stack

- Python 3.11
- FastAPI and Uvicorn
- PyTorch
- OpenCLIP
- Pillow
- NumPy

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API runs at:

```text
http://localhost:8000
```

The first request may take longer while OpenCLIP downloads/loads model weights.

## Configuration

```bash
export SIMILARITY_THRESHOLD=0.75
```

`SIMILARITY_THRESHOLD` controls when a prediction becomes `unknown`.
For prototype testing, CORS is open to all origins.

## API

### Health

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

### Register Reference Image

Register label `A`:

```bash
curl -X POST "http://localhost:8000/register/A" \
  -F "file=@./reference-a.jpg" \
  -F "title=Panel A" \
  -F "description=Reference image for petroglyph panel A."
```

Register label `B`:

```bash
curl -X POST "http://localhost:8000/register/B" \
  -F "file=@./reference-b.png" \
  -F "title=Panel B" \
  -F "description=Reference image for petroglyph panel B."
```

Response:

```json
{
  "success": true,
  "label": "A",
  "title": "Panel A",
  "description": "Reference image for petroglyph panel A."
}
```

Images are saved under `app/data/images/`. Embeddings are stored in `app/data/embeddings.json`.
Titles and descriptions are stored in `app/data/metadata.json`.

### Predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@./scan.webp"
```

Known match:

```json
{
  "prediction": "A",
  "confidence": 0.91,
  "title": "Panel A",
  "description": "Reference image for petroglyph panel A.",
  "scores": {
    "A": 0.91,
    "B": 0.32
  }
}
```

Unknown match:

```json
{
  "prediction": "unknown",
  "confidence": 0.41,
  "title": null,
  "description": null,
  "scores": {
    "A": 0.41,
    "B": 0.38
  }
}
```

## Docker

```bash
docker compose up --build
```

The compose setup exposes port `8000`, mounts the current directory, and runs Uvicorn with hot reload.

## Notes

- Supported uploads: jpg, png, webp.
- Reference titles are limited to 100 characters.
- Reference descriptions are limited to 250 characters.
- Images are converted to RGB, so alpha channels are ignored.
- No database is used.
- Re-registering the same label overwrites its saved embedding, title, and description.
