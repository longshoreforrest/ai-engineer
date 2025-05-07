"""
Low‑latency REST API for quantised object detection.
"""
from fastapi import FastAPI, UploadFile, File
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
import torch, io
from PIL import Image

app = FastAPI(title="Retail Object Detection API")

# ---------- Model -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model   = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval().to(device)

preprocess = weights.transforms()
LABELS = weights.meta["categories"]

# ---------- Endpoint --------------------------------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    batch = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(batch)[0]

    result = [
        {
            "bbox": box.cpu().tolist(),
            "label": LABELS[label],
            "score": float(score),
        }
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"])
        if score > 0.4
    ]
    return {"detections": result}