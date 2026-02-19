# Demo Guide

Two ways to run the perception pipeline without a physical car or ROS installation:

1. **`demo_kitti.py`** — command-line script, outputs an annotated MP4
2. **`app.py`** — Streamlit web app, interactive browser interface

Both run the full pipeline (YOLOv10 + EfficientNet classifiers + EasyOCR) and produce identical annotations.

---

## Prerequisites

```bash
pip install ultralytics torch torchvision opencv-python easyocr
# For the Streamlit app only:
pip install streamlit imageio[ffmpeg]
```

Model weights must be present in `yolov10_ros/models/` (see [models.md](models.md)).

---

## demo_kitti.py — CLI Script

### Input formats

The `--input` argument accepts either:

- **Image sequence folder** — a directory of sequential `.png` / `.jpg` / `.jpeg` files (e.g. a BDD100K tracking sequence or a KITTI `image_02/data/` folder)
- **Video file** — any format supported by OpenCV (`mp4`, `avi`, `mov`, `mkv`, etc.)

### Usage

```bash
cd yolov10_ros

# BDD100K sequence
python3 demo_kitti.py \
    --input /media/brandon/T9/autodrive/BDD100k/track/images/bdd100k/images/track/val/b1c9c847-3bda4659 \
    --output output.mp4

# Video file
python3 demo_kitti.py \
    --input /path/to/dashcam.mp4 \
    --output annotated.mp4

# Limit to first 100 frames (fast test)
python3 demo_kitti.py \
    --input /path/to/sequence \
    --output test.mp4 \
    --limit 100

# CPU-only (no GPU)
python3 demo_kitti.py \
    --input /path/to/sequence \
    --output output.mp4 \
    --device cpu
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Video file or image sequence folder |
| `--output` | `output.mp4` | Output annotated video path |
| `--device` | `0` | CUDA device index or `cpu` |
| `--conf` | `0.3` | Detector confidence threshold |
| `--fps` | `0` | Output FPS (0 = use source FPS for video, 25 for sequences) |
| `--limit` | `0` | Max frames to process (0 = all) |

### Output

A single `.mp4` file with:
- **Gray boxes** — raw detections (car, truck, person, etc.)
- **Colored boxes per state** — traffic lights (green/yellow/red)
- **Blue boxes** — traffic signs (stop, yield, no_left_turn, etc.)
- **Orange boxes** — speed limit signs with parsed value (e.g. "speed limit: 35")
- **Frame counter** in bottom-left corner
- **Per-frame stdout summary** — e.g. `[0042/0203] frame_000042.jpg: light=green_light(0.97), SPEED=35mph`

### BDD100K sequence locations

BDD100K tracking validation sequences live at:
```
/media/brandon/T9/autodrive/BDD100k/track/images/bdd100k/images/track/val/
```

Each subfolder (e.g. `b1c9c847-3bda4659/`) contains ~203 sequential JPEGs. Pick any sequence and pass the full path as `--input`.

---

## app.py — Streamlit App

### Launch

```bash
cd yolov10_ros
streamlit run app.py
```

Opens at `http://localhost:8501` in the browser.

### Interface

The app has two input tabs:

#### Tab 1: Upload video

Drop any `.mp4`, `.avi`, `.mov`, or `.mkv` file using the file uploader. The video is saved to a temp file, decoded with OpenCV, and processed frame by frame.

#### Tab 2: Image sequence folder

Enter the full path to a BDD100K (or KITTI) sequence folder:
```
/media/brandon/T9/autodrive/BDD100k/track/images/bdd100k/images/track/val/b1c9c847-3bda4659
```

The app lists all `.png`/`.jpg`/`.jpeg` files sorted by filename and shows the frame count and resolution before you run the pipeline.

### Sidebar settings

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| Detection confidence | 0.1–0.9 | 0.3 | Detector threshold |
| Output FPS | 5–60 | 25 | Frames per second of the output video |
| Device | GPU / CPU | GPU | Inference device |
| Max frames | 0–∞ | 0 (all) | Limit processing for quick tests |

### Processing

When you press **Run pipeline**:

1. Progress bar appears at the bottom, updated every frame
2. Each frame shows which labels were detected (e.g. "Frame 42/203 — green_light, speed limit: 35")
3. After all frames are processed, the video is re-encoded to H.264 (required for browser playback — OpenCV's native `mp4v` codec is not browser-compatible)
4. The annotated video is embedded directly in the page
5. A **Download annotated video** button lets you save the result

### Model caching

Models are loaded once using `@st.cache_resource` and reused across reruns. Changing the **Device** selector in the sidebar triggers a reload on the next run.

### H.264 re-encoding

After writing all frames with OpenCV (`mp4v` codec), the app runs:
```bash
ffmpeg -vcodec libx264 -crf 23 -preset fast -movflags +faststart output.mp4
```
using the bundled `imageio-ffmpeg` binary (no system `ffmpeg` required). The `+faststart` flag moves the MP4 moov atom to the front of the file, enabling streaming playback before the full file is downloaded.

---

## Annotation Color Reference

| Color | Meaning |
|-------|---------|
| Gray | Raw detector output (car, person, truck, etc.) |
| Green | Green traffic light |
| Red | Red traffic light |
| Yellow/cyan | Yellow traffic light |
| Blue | Traffic sign (non-speed-limit category) |
| Orange | Speed limit sign with OCR value |
| `?` in orange | Speed limit sign found but OCR failed to read the value |

---

## Troubleshooting

### Video plays in app but no annotations visible

The confidence threshold may be too high. Lower it to 0.2 in the sidebar and re-run.

### "No video with supported format and MIME type found"

The H.264 re-encoding step failed. Check that `imageio[ffmpeg]` is installed:
```bash
pip install "imageio[ffmpeg]"
python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

### EasyOCR downloading on first run

Normal. EasyOCR downloads ~100 MB of CRAFT + CRNN weights to `~/.EasyOCR/` on first use. Subsequent runs use the cached files.

### Out of GPU memory

Set **Device** to CPU in the sidebar. EfficientNet B0 and EasyOCR are lightweight; even on CPU a 203-frame sequence completes in a few minutes.

### Folder tab shows 0 images

The folder may not contain `.png`/`.jpg`/`.jpeg` files, or the path has a trailing space. Verify in a terminal:
```bash
ls /path/to/sequence | head
```
