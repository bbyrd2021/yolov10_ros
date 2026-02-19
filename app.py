#!/usr/bin/env python3
"""
Streamlit demo app for the yolov10_ros perception pipeline.

Usage:
    pip install streamlit
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch
import torch.nn.functional as F

# Import helpers from the demo script (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_kitti import (
    load_classifier, preprocess_classifier, preprocess_ocr,
    parse_speed, draw_label, crop_roi,
    LIGHT_CLASSES, SIGN_CLASSES,
    _COLOR_DETECTION, _COLOR_SIGN, _COLOR_SPEED, _COLOR_LIGHT,
    MODEL_DIR,
)

st.set_page_config(page_title="AutoDrive Perception Demo", layout="wide")
st.title("AutoDrive Perception Pipeline")
st.caption("YOLOv10 · EfficientNet B0 classifiers · EasyOCR speed limit reader")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)
    out_fps     = st.slider("Output FPS", 5, 60, 25, 5)
    device_opt  = st.selectbox("Device", ["GPU (cuda:0)", "CPU"], index=0)
    device_str  = "0" if "GPU" in device_opt else "cpu"
    limit       = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=50)


# ── Model loading (cached so they stay in memory across reruns) ────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models(device_str):
    device = torch.device("cuda:0" if device_str.isdigit() else device_str)

    try:
        from ultralytics import YOLOv10
        detector = YOLOv10(str(MODEL_DIR / "yolov10-bdd-vanilla.pt"))
    except ImportError:
        from ultralytics import YOLO
        detector = YOLO(str(MODEL_DIR / "yolov10-bdd-vanilla.pt"))

    light_model = load_classifier(
        MODEL_DIR / "efficientnet_b0_20260211_195116_light_cls.pth",
        len(LIGHT_CLASSES), device
    )
    sign_model = load_classifier(
        MODEL_DIR / "efficientnet_b0_20260202_154542_sign_cls.pth",
        len(SIGN_CLASSES), device
    )

    import easyocr
    reader = easyocr.Reader(['en'], gpu=device_str.isdigit(), verbose=False)

    return detector, light_model, sign_model, reader, device


# ── Main area ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    detector, light_model, sign_model, reader, device = load_models(device_str)

    # Save upload to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if limit:
        total = min(total, int(limit))

    st.info(f"{uploaded.name} — {total} frames at {native_fps:.1f} fps · {w}×{h}")

    if st.button("Run pipeline", type="primary"):
        out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = out_file.name
        out_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(out_fps), (w, h))

        progress = st.progress(0, text="Processing…")
        status   = st.empty()

        for i in range(total):
            ok, image = cap.read()
            if not ok:
                break
            annotated = image.copy()

            with torch.no_grad():
                results = detector.predict(
                    source=image, conf=conf_thresh, imgsz=640,
                    device=device, verbose=False
                )

            boxes = results[0].boxes
            names = results[0].names
            labels_seen = []

            if boxes is not None:
                for j in range(len(boxes)):
                    xyxy  = boxes.xyxy[j].cpu().numpy().astype(int)
                    bconf = float(boxes.conf[j].cpu())
                    cls   = names[int(boxes.cls[j].cpu())]
                    x1, y1, x2, y2 = xyxy

                    crop = crop_roi(image, x1, y1, x2, y2, padding=5)
                    if crop.size == 0:
                        continue

                    if cls == "traffic light":
                        with torch.no_grad():
                            probs = F.softmax(light_model(preprocess_classifier(crop).to(device)), dim=1)
                            lconf, idx = probs.max(dim=1)
                            state = LIGHT_CLASSES.get(int(idx), "unknown")
                        color = _COLOR_LIGHT.get(state, (200, 200, 200))
                        draw_label(annotated, f"{state} {float(lconf):.2f}", x1, y1, x2, y2, color)
                        labels_seen.append(state)

                    elif cls == "traffic sign":
                        with torch.no_grad():
                            probs = F.softmax(sign_model(preprocess_classifier(crop).to(device)), dim=1)
                            sconf, idx = probs.max(dim=1)
                            sign_type = SIGN_CLASSES.get(int(idx), "unknown")
                        if sign_type == "speed_limit":
                            speed, oconf = parse_speed(
                                reader.readtext(preprocess_ocr(crop), allowlist='0123456789', detail=1)
                            )
                            if speed:
                                draw_label(annotated, f"speed limit: {speed}", x1, y1, x2, y2, _COLOR_SPEED)
                                labels_seen.append(f"speed limit: {speed}")
                            else:
                                draw_label(annotated, "speed limit: ?", x1, y1, x2, y2, _COLOR_SPEED)
                        else:
                            draw_label(annotated, f"{sign_type} {float(sconf):.2f}", x1, y1, x2, y2, _COLOR_SIGN)
                            labels_seen.append(sign_type)
                    else:
                        draw_label(annotated, f"{cls} {bconf:.2f}", x1, y1, x2, y2, _COLOR_DETECTION)

            cv2.putText(annotated, f"{i+1}/{total}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(annotated)

            pct = (i + 1) / total
            progress.progress(pct, text=f"Frame {i+1}/{total} — {', '.join(labels_seen) or 'no detections'}")

        cap.release()
        writer.release()
        progress.empty()
        status.success("Done!")

        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download annotated video", f, file_name="annotated_output.mp4", mime="video/mp4")
