# AI Engineer Assessment — Reference Solution

This README explains how to **build, run and test** the reference implementation delivered for the *Technological Assessment for AI Engineer Candidates*.

---

## 1  Repository Layout

```text
ai-engineer-solution/
├── docker/
│   ├── base_api/                 # FastAPI container (simple REST)
│   └── triton_adapter/           # PyTriton Python‑backend façade
├── model_repository/             # Triton model store (TorchScript)
│   └── retail_detector/
│       ├── 1/                    # version 1 of the model
│       │   ├── model.pt
│       └── config.pbtxt
├── compose/
│   └── docker-compose.yml        # Triton + adapter orchestration
├── app/                          # FastAPI service code
├── scripts/                      # Smoke‑test clients
├── presentation/                 # End‑to‑end scenario outline
├── api_test.ipynb                # Interactive notebook client
└── README.md                     # (this file)
```

---

## 2  Prerequisites

| Component | Version (tested) | Notes |
|-----------|------------------|-------|
| **Docker** | ≥ 24.0 | Required for all paths |
| **Python** | 3.9 – 3.11 | Needed for smoke‑test scripts / notebook |

---

## 3  Quick Start

### 3.1 Clone the repo

```bash
git clone <YOUR_FORK_URL> ai-engineer
cd ai-engineer
```

---

## 4  Path #1 — FastAPI REST service (PoC)

### 4.1 Build & run

```bash
# Build the image
DOCKER_BUILDKIT=1 docker build -t retail-detector-api -f docker/base_api/Dockerfile .

# Launch the container on port 8000
docker run --rm -p 8000:8000 retail-detector-api
```
When the container starts **for the very first time** it downloads the `ssdlite320_mobilenet_v3_large_coco_2023.pth` weights

### 4.2 Smoke test (script)

Install packages if needed:
```bash
pip install -r scripts/requirements.txt 
```

Run the script to identify a person:
```bash
python scripts/test_rest_client.py scripts/retail_person.jpeg
```

Run the script for another object type:
```bash
python scripts/test_rest_client.py scripts/horse.jpeg
```

The script POSTs the image to `http://localhost:8000/predict` and prints the JSON detections.

---

## 5  Path #2 — Triton Inference Server (production‑grade)

### 5.1 Build & launch the stack

```bash
cd compose
docker compose up --build    # first boot ~2 GB download for Triton image
```
The stack spins up two services:

* **triton:** NVIDIA Triton server exposing HTTP :8000 + gRPC :8001
* **adapter:** lightweight Python service translating Triton output (optional)

### 5.2 Smoke test

In a *new* terminal, with the stack still running:

```bash
python scripts/test_triton_client.py scripts/retail_person.jpeg
```

To identify another object type:
```bash
python scripts/test_triton_client.py scripts/horse.jpeg
```

The script uses the Triton HTTP client to send a tensor, then prints the response and decoded detections.

---

## 6  Directory‑by‑Directory Guide

| Directory | Key files | Purpose |
|-----------|-----------|---------|
| `docker/base_api/` | `Dockerfile`, `requirements.txt` | Minimal REST wrapper around quantised TorchVision model |
| `docker/triton_adapter/` | `Dockerfile`, `inference_adapter.py` | Registers a PyTriton Python backend for post‑processing |
| `model_repository/…` | `model.pt`, `config.pbtxt` | TorchScript model + Triton config (batch size, IO tensors) |
| `compose/` | `docker-compose.yml` | Glue the Triton server and adapter together |
| `scripts/` | `test_rest_client.py`, `test_triton_client.py` | CLI smoke tests |
| `presentation/` | `end_to_end_scenario.md` | Slides outline for interview discussion |

---

## 7  Extending / Customising

* **Change the backbone** – Swap out the model in `app/main.py` and retrain; update `model_repository/` with the new TorchScript artefact.
* **Autoscaling** – Deploy Triton + adapter to Kubernetes and add Horizontal Pod Autoscaler based on GPU utilisation.
* **Monitoring** – Expose Triton metrics on :8002 and scrape with Prometheus + Grafana dashboards.

---
## 8 Model Architecture

- **Backbone:** MobileNet V3‑Large  
  - Lightweight CNN for mobile/edge.  
  - Lower accuracy than big nets (ResNet, EfficientNet) but much faster and smaller.  

- **Detection head:** SSDLite‑320  
  - Single‑Shot Detector predicts boxes and class scores in one pass.  
  - “Lite” uses depthwise‑separable convolutions for efficiency.  
  - Input resized to 320×320 pixels.  

- **Dataset:** COCO (Common Objects in Context)  
  - 80 object categories (person, car, dog, etc.) with rich bounding‑box annotations.  
  - Benchmark widely used for object‑detection research.  
  - More info: https://cocodataset.org/  

> **Overall:** SSDLite320 + MobileNet V3‑Large yields very low latency on CPU/GPU, at the cost of some accuracy compared to heavier detectors.  

---

## 9  Alternative Architectures Comparison

| Use‑case                  | Model (Selected)                          | Pros                                      | Cons                                    |
|---------------------------|-------------------------------------------|-------------------------------------------|-----------------------------------------|
| **Current baseline**      | **SSDLite320 MobileNet V3‑Large**         | Low latency; small memory footprint; easy to deploy on CPU/GPU | Moderate accuracy, especially on small objects |
| Higher accuracy           | Faster R‑CNN w/ ResNet50                  | Better detection, especially small or crowded objects | Slower, heavier memory footprint        |
| Real‑time on GPU/edge     | YOLOv5 / YOLOv8                           | Extremely fast; simple deployment; high FPS | Slightly more complex training pipeline |
| Balanced speed & quality  | EfficientDet‑D0 or D1                     | Better accuracy per FLOP than SSD variants; scalable | Moderate latency                        |
| Transformer‑based research| DETR                                      | End‑to‑end without anchors; models relations | High latency; needs large data to train |
| Ultra‑tiny on CPU/mobile  | Tiny‑YOLO / NanoDet                       | Very low latency on CPU; tiny binary size | Lower accuracy                          |

---

## 10  Attribution & Licence

* Base weights: *TorchVision* (Apache 2.0)
* Triton Inference Server: *NVIDIA*, licence SPDX‑License‑Identifier: BSD‑3‑Clause

This reference implementation itself is released under the **MIT Licence**. See `LICENSE` file.

## 11  GenAI Disclosure

As required by the assignment, generative AI tools were used during the development of this solution:

* O3 was used to create the preliminary structure of this README and initial scaffolding of the codebase.

* o4-mini was used to iterate and refine the implementation, including API definitions, Docker configuration, and test scripts.


