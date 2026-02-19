#!/usr/bin/env python3
"""
Standalone KITTI demo — no ROS required.

Runs the full perception pipeline (YOLOv10 detector + EfficientNet classifiers
+ EasyOCR speed limit reader) on a KITTI image sequence and writes an
annotated output video.

Usage:
    python3 demo_kitti.py --sequence /path/to/kitti/image_02/data/ \
                          --output output.mp4 \
                          --device 0

The --sequence directory should contain a flat set of PNG/JPG frames,
as downloaded from the KITTI raw data benchmark (image_02 = left color camera).
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"  # yolov10_ros/models/ (symlinked to project models/)

# ── Valid US speed limits ──────────────────────────────────────────────────────
_VALID_SPEEDS = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}

# ── EfficientNet preprocessing ─────────────────────────────────────────────────
_INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Class mappings ─────────────────────────────────────────────────────────────
LIGHT_CLASSES = {
    0: "green_left", 1: "green_light", 2: "red_left",
    3: "red_light",  4: "yellow_left", 5: "yellow_light",
}
SIGN_CLASSES = {
    0: "detour",          1: "do_not_enter",    2: "go_straight_only",
    3: "no_left_turn",    4: "no_right_turn",   5: "no_straight",
    6: "no_u_turn",       7: "pedestrian_crossing", 8: "railroad_crossing",
    9: "roadwork",       10: "speed_limit",     11: "stop",
    12: "turn_left_only",13: "turn_right_only", 14: "yield",
}

# ── Annotation colors (BGR) ────────────────────────────────────────────────────
_COLOR_DETECTION  = (180, 180, 180)  # gray  — raw detections
_COLOR_SIGN       = (200, 100,   0)  # blue  — traffic signs
_COLOR_SPEED      = ( 30, 140, 255)  # orange — speed limit
_COLOR_LIGHT = {
    "green_light":  ( 40, 200,  40),
    "green_left":   ( 40, 200,  40),
    "red_light":    ( 40,  40, 220),
    "red_left":     ( 40,  40, 220),
    "yellow_light": ( 30, 200, 220),
    "yellow_left":  ( 30, 200, 220),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def crop_roi(image, xmin, ymin, xmax, ymax, padding=0):
    h, w = image.shape[:2]
    x1 = max(0, xmin - padding)
    y1 = max(0, ymin - padding)
    x2 = min(w, xmax + padding)
    y2 = min(h, ymax + padding)
    return image[y1:y2, x1:x2]


def preprocess_classifier(crop):
    img = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def preprocess_ocr(crop, min_height=64):
    h, w = crop.shape[:2]
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def draw_label(image, text, x1, y1, x2, y2, color):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    ty = max(y1 - 4, th + bl)
    cv2.rectangle(image, (x1, ty - th - bl - 2), (x1 + tw, ty), color, -1)
    cv2.putText(image, text, (x1, ty - bl - 1), font, scale, (0, 0, 0), thick)


def load_classifier(weights_path, num_classes, device):
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_speed(ocr_results, conf_threshold=0.3):
    best, best_conf = None, 0.0
    for (_, text, conf) in ocr_results:
        if conf < conf_threshold:
            continue
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            continue
        try:
            val = int(digits)
        except ValueError:
            continue
        if val in _VALID_SPEEDS and conf > best_conf:
            best, best_conf = val, conf
    return best, best_conf


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KITTI perception demo")
    parser.add_argument("--sequence", required=True,
                        help="Path to KITTI image sequence folder (image_02/data/)")
    parser.add_argument("--output", default="output.mp4",
                        help="Output video path")
    parser.add_argument("--device", default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detector confidence threshold")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Output video FPS (default 25 plays KITTI at ~2.5x speed)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max frames to process (0 = all)")
    args = parser.parse_args()

    device = torch.device(
        "cuda:{}".format(args.device) if args.device.isdigit() else args.device
    )
    print("Device:", device)

    # Collect frames
    seq_path = Path(args.sequence)
    frames = sorted(
        p for p in seq_path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frames:
        print("ERROR: No images found in", seq_path)
        sys.exit(1)
    if args.limit:
        frames = frames[:args.limit]
    print(f"Found {len(frames)} frames")

    # Load detector
    print("Loading YOLOv10 detector...")
    try:
        from ultralytics import YOLOv10
        detector = YOLOv10(str(MODEL_DIR / "yolov10-bdd-vanilla.pt"))
    except ImportError:
        from ultralytics import YOLO
        detector = YOLO(str(MODEL_DIR / "yolov10-bdd-vanilla.pt"))
    print(f"  Detector classes: {list(detector.names.values())}")

    # Load classifiers
    print("Loading EfficientNet classifiers...")
    light_model = load_classifier(
        MODEL_DIR / "efficientnet_b0_20260211_195116_light_cls.pth",
        len(LIGHT_CLASSES), device
    )
    sign_model = load_classifier(
        MODEL_DIR / "efficientnet_b0_20260202_154542_sign_cls.pth",
        len(SIGN_CLASSES), device
    )
    print("  Classifiers ready")

    # Load EasyOCR
    print("Loading EasyOCR (may download models on first run)...")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=args.device.isdigit(), verbose=False)
    print("  EasyOCR ready")

    # Video writer — use first frame for dimensions
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    print(f"\nProcessing {len(frames)} frames -> {args.output}\n")

    for i, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        annotated = image.copy()

        # ── Detector ──
        with torch.no_grad():
            results = detector.predict(
                source=image, conf=args.conf, imgsz=640,
                device=device, verbose=False
            )

        boxes = results[0].boxes
        names = results[0].names
        detections = []
        if boxes is not None:
            for j in range(len(boxes)):
                xyxy = boxes.xyxy[j].cpu().numpy().astype(int)
                conf = float(boxes.conf[j].cpu())
                cls  = names[int(boxes.cls[j].cpu())]
                detections.append((cls, conf, xyxy))

        frame_summary = []

        for cls, conf, xyxy in detections:
            x1, y1, x2, y2 = xyxy
            crop = crop_roi(image, x1, y1, x2, y2, padding=5)
            if crop.size == 0:
                continue

            if cls == "traffic light":
                # ── Light classifier ──
                with torch.no_grad():
                    tensor = preprocess_classifier(crop).to(device)
                    probs = F.softmax(light_model(tensor), dim=1)
                    lconf, idx = probs.max(dim=1)
                    state = LIGHT_CLASSES.get(int(idx), "unknown")
                    lconf = float(lconf)
                color = _COLOR_LIGHT.get(state, (200, 200, 200))
                label = f"{state} {lconf:.2f}"
                draw_label(annotated, label, x1, y1, x2, y2, color)
                frame_summary.append(f"light={state}({lconf:.2f})")

            elif cls == "traffic sign":
                # ── Sign classifier ──
                with torch.no_grad():
                    tensor = preprocess_classifier(crop).to(device)
                    probs = F.softmax(sign_model(tensor), dim=1)
                    sconf, idx = probs.max(dim=1)
                    sign_type = SIGN_CLASSES.get(int(idx), "unknown")
                    sconf = float(sconf)

                if sign_type == "speed_limit":
                    # ── OCR ──
                    ocr_crop = preprocess_ocr(crop)
                    ocr_results = reader.readtext(
                        ocr_crop, allowlist='0123456789', detail=1
                    )
                    speed, oconf = parse_speed(ocr_results)
                    if speed is not None:
                        label = f"speed limit: {speed} ({oconf:.2f})"
                        draw_label(annotated, label, x1, y1, x2, y2, _COLOR_SPEED)
                        frame_summary.append(f"SPEED={speed}mph")
                    else:
                        draw_label(annotated, "speed limit: ?", x1, y1, x2, y2, _COLOR_SPEED)
                        frame_summary.append("speed_limit(ocr_fail)")
                else:
                    label = f"{sign_type} {sconf:.2f}"
                    draw_label(annotated, label, x1, y1, x2, y2, _COLOR_SIGN)
                    frame_summary.append(f"sign={sign_type}({sconf:.2f})")

            else:
                # Raw detection
                label = f"{cls} {conf:.2f}"
                draw_label(annotated, label, x1, y1, x2, y2, _COLOR_DETECTION)

        # Frame counter overlay
        cv2.putText(annotated, f"frame {i+1}/{len(frames)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(annotated)

        summary = ", ".join(frame_summary) if frame_summary else "no detections"
        print(f"  [{i+1:04d}/{len(frames)}] {frame_path.name}: {summary}")

    writer.release()
    print(f"\nDone. Output written to: {args.output}")


if __name__ == "__main__":
    main()
