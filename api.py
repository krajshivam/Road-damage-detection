"""
Road Damage Detection — FastAPI Inference Server
Run: uv run uvicorn api:app --reload

Endpoints:
    GET  /         - API info
    GET  /health   - Health check
    POST /predict  - Run inference on uploaded image
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import time

# Initialize FastAPI app
app = FastAPI(
    title="Road Damage Detection API",
    description="YOLOv8s fine-tuned on RDD2022 — detects 7 types of road damage",
    version="1.0.0",
)

# Load model once at startup — not on every request
# Why: loading model takes ~1 second. If we loaded per request,
# every API call would be 1 second slower. Load once, reuse forever.
MODEL_PATH = "runs/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model failed to load: {e}")
    model = None

CLASS_NAMES = [
    "Alligator Cracks",
    "Damaged crosswalk",
    "Damaged paint",
    "Longitudinal Cracks",
    "Manhole cover",
    "Potholes",
    "Transverse Cracks",
]


@app.get("/")
def root():
    """API information endpoint."""
    return {
        "name": "Road Damage Detection API",
        "version": "1.0.0",
        "model": MODEL_PATH,
        "classes": CLASS_NAMES,
        "status": "ready" if model else "model not loaded",
    }


@app.get("/health")
def health():
    """Health check — confirms server is running."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    """
    Detect road damage in uploaded image.

    Args:
        file: Road image (JPEG or PNG)
        conf: Confidence threshold (default 0.25)

    Returns:
        JSON with detections, confidence scores, bounding boxes
    """
    # Check model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"File must be an image. Got: {file.content_type}"
        )

    # Read uploaded file into memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference and measure latency
    start = time.time()
    results = model.predict(source=image, conf=conf, verbose=False)
    latency_ms = round((time.time() - start) * 1000, 2)

    # Parse detections from results
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                }
            )

    return JSONResponse(
        {
            "filename": file.filename,
            "image_size": {"width": image.width, "height": image.height},
            "detections": detections,
            "total_detections": len(detections),
            "latency_ms": latency_ms,
            "conf_threshold": conf,
        }
    )
