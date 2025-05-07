#!/usr/bin/env python3
import sys
import json

import cv2
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

# COCO class names (index 1 is “person”)
COCO_LABELS = {
    0: '__background__',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

def preprocess(image_path, size=(320, 320)):
    """Load image, convert BGR→RGB, resize, normalize, transpose to CHW, add batch dim."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size).astype(np.float32) / 255.0
    tensor = np.transpose(img_resized, (2, 0, 1))[None, ...]
    return tensor, (w0, h0), img_bgr

def postprocess_and_draw(orig_bgr, detections, orig_size, output_path):
    """Draw boxes and class labels on image and save to disk."""
    w0, h0 = orig_size
    det = detections.reshape(-1, 6)
    for (x1, y1, x2, y2, score, cls) in det:
        cls_i = int(cls)
        label = COCO_LABELS.get(cls_i, f"cls{cls_i}")
        # scale coords from 320→orig
        x1i = int(x1 * w0 / 320)
        x2i = int(x2 * w0 / 320)
        y1i = int(y1 * h0 / 320)
        y2i = int(y2 * h0 / 320)
        cv2.rectangle(orig_bgr, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
        cv2.putText(orig_bgr, f"{label}:{score:.2f}", (x1i, max(0,y1i-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(output_path, orig_bgr)
    print(f"Wrote visualization to {output_path}")

def main():
    if len(sys.argv) not in (2,3):
        print(f"Usage: {sys.argv[0]} <input.jpg> [<output.jpg>]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv)==3 else "out.jpg"

    # Preprocess
    input_tensor, orig_size, orig_bgr = preprocess(img_path)

    # Triton client
    client = InferenceServerClient(url="localhost:8000")
    inp = InferInput("INPUT__0", input_tensor.shape, "FP32")
    inp.set_data_from_numpy(input_tensor)
    out_req = InferRequestedOutput("OUTPUT__0", binary_data=False)

    # Inference
    resp = client.infer(model_name="retail_detector",
                        inputs=[inp],
                        outputs=[out_req])

    # Extract detections
    dets = resp.as_numpy("OUTPUT__0")
    print("Detections (x1,y1,x2,y2,score,class):")
    for det in dets.reshape(-1,6):
        x1,y1,x2,y2,score,cls = det
        cls_i = int(cls)
        label = COCO_LABELS.get(cls_i, f"cls{cls_i}")
        print(f" - {label:10s} @ [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], score {score:.2f}")

    # Draw and save
    postprocess_and_draw(orig_bgr, dets, orig_size, out_path)

if __name__ == "__main__":
    main()
