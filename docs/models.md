# Models Reference

Three learned models are used in the pipeline. All three are loaded from files in the `models/` directory (symlinked to the project-level `models/` folder; path is not committed to git).

---

## 1. YOLOv10 Detector

| Property | Value |
|----------|-------|
| Architecture | YOLOv10 (NMS-free, one-to-one head) |
| Training dataset | BDD100K (driving scenes, daytime + nighttime) |
| Input resolution | 640 × 640 (letterboxed, handled by ultralytics) |
| Output | Bounding boxes in original pixel coordinates |
| File | `yolov10-bdd-vanilla.pt` |
| Loading | `ultralytics.YOLOv10` or `ultralytics.YOLO` (auto-fallback) |

### BDD100K Classes (10 total)

| Index | Class | Notes |
|-------|-------|-------|
| — | pedestrian | Person on foot |
| — | rider | Person on a bicycle or motorcycle |
| — | car | Passenger vehicles |
| — | truck | Large freight vehicles |
| — | bus | Transit buses |
| — | train | Trains on track |
| — | motorcycle | Motorcycles |
| — | bicycle | Bicycles |
| — | traffic light | → forwarded to light classifier |
| — | traffic sign | → forwarded to sign classifier |

*Class indices are determined by the model's internal `names` dict and may vary across training runs. The node uses `results[0].names[cls_idx]` — never hardcoded indices.*

### Why YOLOv10?

YOLOv10 replaces the traditional post-NMS step with a dual-assignment training strategy that produces exactly one prediction per object. This:
- Eliminates the non-maximum suppression hyperparameter (IoU threshold)
- Reduces end-to-end latency by ~20–30% compared to equivalently-sized YOLOv8
- Produces cleaner bounding boxes on overlapping objects (e.g. traffic lights on a gantry)

### Loading via ultralytics

The node tries `YOLOv10` first for compatibility with pinned ultralytics versions, and falls back to the generic `YOLO` class:

```python
try:
    from ultralytics import YOLOv10
    detector = YOLOv10("models/yolov10-bdd-vanilla.pt")
except ImportError:
    from ultralytics import YOLO
    detector = YOLO("models/yolov10-bdd-vanilla.pt")
```

`model.predict()` handles all preprocessing internally (resize, normalize, letterbox) and returns results in the **original image pixel coordinate frame** — no inverse scaling needed.

---

## 2. Traffic Light Classifier

| Property | Value |
|----------|-------|
| Architecture | EfficientNet B0 |
| Training dataset | Cropped traffic light images |
| Input size | 224 × 224 (standard ImageNet convention) |
| Output classes | 6 |
| File | `efficientnet_b0_20260211_195116_light_cls.pth` |
| Checkpoint format | `{"epoch": int, "model_state_dict": OrderedDict, ...}` |

### Class Mapping

The class indices are assigned alphabetically by the `torchvision.datasets.ImageFolder` loader during training. **The order is fixed by the training folder names.**

| Index | Class | Meaning |
|-------|-------|---------|
| 0 | `green_left` | Green left-arrow signal |
| 1 | `green_light` | Solid green (go) |
| 2 | `red_left` | Red left-arrow signal |
| 3 | `red_light` | Solid red (stop) |
| 4 | `yellow_left` | Yellow left-arrow signal |
| 5 | `yellow_light` | Solid yellow (caution) |

### Preprocessing

```
Crop (arbitrary size, BGR)
  → cv2.resize(224, 224)
  → BGR → RGB
  → scale to [0, 1]
  → subtract ImageNet mean [0.485, 0.456, 0.406]
  → divide by ImageNet std  [0.229, 0.224, 0.225]
  → HWC → CHW → add batch dim
  → tensor (1, 3, 224, 224) on GPU
```

### Loading Pattern

The `.pth` file is a **training checkpoint dict**, not a serialised model. You cannot call `model = torch.load(...)` directly. The correct loading sequence:

```python
from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights=None)                          # random init
model.classifier[1] = torch.nn.Linear(1280, 6)                # replace head
ckpt = torch.load("efficientnet_b0_..._light_cls.pth",        # load dict
                  map_location=device)
model.load_state_dict(ckpt["model_state_dict"])                # restore weights
model.eval()
```

EfficientNet B0 has 1280 features before the classifier head (`model.classifier[1].in_features == 1280`).

---

## 3. Traffic Sign Classifier

| Property | Value |
|----------|-------|
| Architecture | EfficientNet B0 |
| Training dataset | Cropped US traffic sign images |
| Input size | 224 × 224 |
| Output classes | 15 |
| File | `efficientnet_b0_20260202_154542_sign_cls.pth` |
| Checkpoint format | Same as light classifier |

### Class Mapping

| Index | Class | Category |
|-------|-------|----------|
| 0 | `detour` | Warning |
| 1 | `do_not_enter` | Regulatory |
| 2 | `go_straight_only` | Regulatory |
| 3 | `no_left_turn` | Regulatory |
| 4 | `no_right_turn` | Regulatory |
| 5 | `no_straight` | Regulatory |
| 6 | `no_u_turn` | Regulatory |
| 7 | `pedestrian_crossing` | Warning |
| 8 | `railroad_crossing` | Warning |
| 9 | `roadwork` | Warning |
| 10 | `speed_limit` | Regulatory — numeric value read by OCR node |
| 11 | `stop` | Regulatory |
| 12 | `turn_left_only` | Regulatory |
| 13 | `turn_right_only` | Regulatory |
| 14 | `yield` | Regulatory |

### Loading Pattern

Identical to the light classifier; only the number of output classes differs:

```python
model.classifier[1] = torch.nn.Linear(1280, 15)
```

---

## Model File Naming Convention

The `.pth` files embed a timestamp in their filename:

```
efficientnet_b0_YYYYMMDD_HHMMSS_<task>_cls.pth
```

This allows multiple training runs to coexist in `models/` without overwriting each other. The launch file hardcodes the specific filename, so to swap in a newer checkpoint, update the `light_weights` / `sign_weights` argument in `pipeline.launch`.

---

## Adding the Model Files

Model weights are not committed to git (they are gitignored via `models/`). Place them directly in `yolov10_ros/models/`:

```
yolov10_ros/models/
├── yolov10-bdd-vanilla.pt
├── efficientnet_b0_20260211_195116_light_cls.pth
└── efficientnet_b0_20260202_154542_sign_cls.pth
```

The `models/` directory is a symlink to the project-level `models/` folder so weights are shared across scripts and are not duplicated.
