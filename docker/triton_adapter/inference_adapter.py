"""
Registers a Python backend with Triton that forwards the request
to the built‑in PyTorch model and returns JSON‑serialisable output.
"""

import numpy as np
from PIL import Image
import torch, torchvision
from pytriton.decorators import batch, decoupled, triton
from pytriton.models.base import TritonPythonModel

CLASS_NAMES = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]

@triton(model_name="retail_detector", outputs=[("detections", np.float32, (-1, 6))])
class RetailDetector(TritonPythonModel):
    def initialize(self, args):
        weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights).eval()
        self.pre = weights.transforms()

    @batch
    def execute(self, images):
        """
        `images` arrives as NHWC FP32 0‑1. Return numpy array with
        [x1, y1, x2, y2, score, label_id] per detection.
        """
        with torch.no_grad():
            tensor_batch = torch.from_numpy(images).permute(0, 3, 1, 2)
            out = self.model(tensor_batch)
        res = []
        for det in out:
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            keep = scores > 0.4
            res.append(np.column_stack([boxes[keep], scores[keep], labels[keep]]))
        return (np.array(res, dtype=np.float32),)
