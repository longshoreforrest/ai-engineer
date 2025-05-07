import torch, torchvision, pathlib

class Wrapped(torch.nn.Module):
    """
    TorchScript wrapper for SSDLite.
    Expects  [N, 3, 320, 320]  (N==1 in Triton, FP32 0‑1).
    Returns  [M, 6]  (x1, y1, x2, y2, score, class_id)
    """
    def __init__(self, thr: float = 0.4):
        super().__init__()
        w = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.det = (
            torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=w).eval()
        )
        self.thr = thr

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Triton sends batch shape [1, 3, 320, 320]; take first image
        img = batch[0]                              # [3, 320, 320]
        det = self.det([img])[0]                    # dict
        keep   = det["scores"] > self.thr
        boxes  = det["boxes"][keep]
        scores = det["scores"][keep][:, None]
        labels = det["labels"][keep][:, None].float()
        return torch.cat([boxes, scores, labels], 1)  # [M, 6]

# ── export ───────────────────────────────────────────────────────────
repo = pathlib.Path("model_repository/retail_detector/1")
repo.mkdir(parents=True, exist_ok=True)
torch.jit.script(Wrapped()).save(repo / "model.pt")
print("✅ TorchScript wrapper saved →", repo / "model.pt")
