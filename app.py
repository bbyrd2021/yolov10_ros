#!/usr/bin/env python3
"""
Streamlit demo app for the yolov10_ros perception pipeline.

Usage:
    pip install streamlit
    streamlit run app.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch
import torch.nn.functional as F

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


# ── Model loading (cached across reruns) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
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


# ── Shared pipeline runner ─────────────────────────────────────────────────────
def run_pipeline(frames, total, w, h):
    """Process an iterable of BGR frames. Yields annotated frames."""
    detector, light_model, sign_model, reader, device = load_models(device_str)

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(out_fps), (w, h))

    progress = st.progress(0, text="Processing…")
    status = st.empty()

    for i, image in enumerate(frames):
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
                        speed, _ = parse_speed(
                            reader.readtext(preprocess_ocr(crop), allowlist='0123456789', detail=1)
                        )
                        label = f"speed limit: {speed}" if speed else "speed limit: ?"
                        draw_label(annotated, label, x1, y1, x2, y2, _COLOR_SPEED)
                        if speed:
                            labels_seen.append(f"speed limit: {speed}")
                    else:
                        draw_label(annotated, f"{sign_type} {float(sconf):.2f}", x1, y1, x2, y2, _COLOR_SIGN)
                        labels_seen.append(sign_type)
                else:
                    draw_label(annotated, f"{cls} {bconf:.2f}", x1, y1, x2, y2, _COLOR_DETECTION)

        cv2.putText(annotated, f"{i+1}/{total}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(annotated)
        progress.progress((i + 1) / total,
                          text=f"Frame {i+1}/{total} — {', '.join(labels_seen) or 'no detections'}")

    writer.release()

    # Re-encode to H.264 for browser playback
    progress.progress(1.0, text="Re-encoding to H.264…")
    h264_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_path = h264_file.name
    h264_file.close()

    import imageio_ffmpeg
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", out_path,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-movflags", "+faststart", h264_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    progress.empty()
    status.success("Done!")
    st.video(h264_path)
    with open(h264_path, "rb") as f:
        st.download_button("Download annotated video", f,
                           file_name="annotated_output.mp4", mime="video/mp4")


# ── Input tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_folder = st.tabs(["Upload video", "Image sequence folder"])

# ── Tab 1: video file upload ───────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Drop a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            input_path = tmp.name

        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if limit:
            total = min(total, int(limit))

        st.info(f"{uploaded.name} — {total} frames · {fps_src:.1f} fps source · {w}×{h}")

        if st.button("Run pipeline", type="primary", key="run_upload"):
            def _video_frames(cap, total):
                for _ in range(total):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame
                cap.release()

            run_pipeline(_video_frames(cap, total), total, w, h)

# ── Tab 2: image sequence folder ───────────────────────────────────────────────
with tab_folder:
    folder_path = st.text_input(
        "Sequence folder path",
        placeholder="/media/brandon/T9/autodrive/BDD100k/track/images/bdd100k/images/track/val/<sequence>",
    )

    if folder_path:
        p = Path(folder_path)
        if not p.is_dir():
            st.error(f"Directory not found: {folder_path}")
        else:
            image_files = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not image_files:
                st.error("No images found in that folder.")
            else:
                total = len(image_files)
                if limit:
                    image_files = image_files[:int(limit)]
                    total = len(image_files)

                sample = cv2.imread(str(image_files[0]))
                h, w = sample.shape[:2]

                st.info(f"{p.name} — {total} frames · {w}×{h}")

                if st.button("Run pipeline", type="primary", key="run_folder"):
                    def _folder_frames(files):
                        for fp in files:
                            frame = cv2.imread(str(fp))
                            if frame is not None:
                                yield frame

                    run_pipeline(_folder_frames(image_files), total, w, h)
