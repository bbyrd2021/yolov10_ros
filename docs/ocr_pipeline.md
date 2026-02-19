# Speed Limit OCR Pipeline

Detailed walkthrough of how the `speed_limit_ocr_node` reads numeric values off speed limit sign crops.

---

## Why OCR Instead of Another Classifier?

The sign classifier identifies **what type** of sign is present (one of 15 categories). For most sign types — stop, yield, pedestrian crossing — the type label is all the autonomy stack needs.

Speed limit signs are different: the critical information is the **number**, not the category. A single "speed_limit" output from the classifier could be 25, 35, 55, or 80 mph. Training a classifier to predict the specific speed value would require:

- Separate training data for each speed value
- A fixed vocabulary (can't generalise to unusual limits like 15 or 85)
- Retraining every time a new speed value needs to be supported

OCR generalises naturally: the same CRAFT + CRNN model reads any digit sequence, and the whitelist validation constrains results to known-good values without retraining.

---

## System Overview

```
ClassifiedSign (sign_type == "speed_limit")
  │
  ├─ xmin, ymin, xmax, ymax  ──→  crop_roi(image, +10px padding)
  │                                     │
  │                                     ▼
  │                               ┌──────────────┐
  │                               │ Preprocessing │
  │                               │  1. Upscale   │
  │                               │  2. Grayscale │
  │                               │  3. CLAHE     │
  │                               └──────┬───────┘
  │                                      │
  │                                      ▼
  │                               ┌──────────────┐
  │                               │   EasyOCR    │
  │                               │  readtext()  │
  │                               │  allowlist=  │
  │                               │  '0123456789'│
  │                               └──────┬───────┘
  │                                      │
  │                               [(bbox, text, conf), ...]
  │                                      │
  │                                      ▼
  │                               ┌──────────────┐
  │                               │  parse_speed │
  │                               │  conf filter │
  │                               │  digit strip │
  │                               │  int cast    │
  │                               │  whitelist   │
  │                               └──────┬───────┘
  │                                      │
  └─ bounding box  ─────────────→  SpeedLimit message
```

---

## Step 1: Crop

The bounding box from the sign classifier is expanded by `roi_padding` (default: 10 px) on all sides before cropping. A wider crop than the classifier default (5 px) is used because:

1. EasyOCR's CRAFT detector uses the surrounding context to locate text regions.
2. The red border of a speed limit sign acts as a boundary anchor that helps CRAFT find the text area.
3. An overly tight crop that clips the numeral strokes causes the CRNN recogniser to fail.

```python
crop = crop_roi(image, sign.xmin, sign.ymin, sign.xmax, sign.ymax, padding=10)
```

---

## Step 2: Upscale

EasyOCR's CRNN text recogniser requires the character height to be at least ~30–50 px for reliable recognition. Signs detected far away may produce crops as small as 20–30 px tall.

If the crop height is below `min_crop_height` (default: 64 px), it is upscaled proportionally using **bicubic interpolation**:

```python
if h < min_crop_height:
    scale = min_crop_height / h
    crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)
```

Bicubic is preferred over bilinear for small-to-large upscaling because it preserves sharpness by considering a 4×4 neighbourhood of pixels.

---

## Step 3: Grayscale + CLAHE

```python
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
processed = clahe.apply(gray)
```

**Why grayscale?** CRAFT (the EasyOCR text detection backbone) operates on single-channel images internally. Converting to grayscale before passing to EasyOCR avoids a redundant internal conversion.

**CLAHE (Contrast Limited Adaptive Histogram Equalisation)** addresses two real-world problems:

1. **Sun fading** — older signs lose pigment saturation over time, reducing the contrast between the black numerals and the white sign face.
2. **Shadows** — a tree or overpass casting a partial shadow can darken one half of a sign unevenly, making the CRNN misidentify characters.

Standard histogram equalisation would amplify noise uniformly. CLAHE divides the image into tiles (`tileGridSize=(4, 4)`) and equalises each tile independently, then bilinearly blends tile boundaries. The `clipLimit=2.0` cap prevents noise amplification in nearly-uniform regions (e.g. the white sign background).

---

## Step 4: EasyOCR

```python
ocr_results = reader.readtext(
    processed,
    allowlist='0123456789',
    detail=1
)
```

`reader` is an `easyocr.Reader(['en'])` instance created once at node startup.

**EasyOCR internals:**

1. **CRAFT (Character Region Awareness for Text Detection)** — a CNN that produces a character-level heatmap and affinity map. Text regions are segmented from these heatmaps.
2. **CRNN (Convolutional Recurrent Neural Network)** — processes each detected text region through a VGG-style CNN feature extractor followed by a BiLSTM, and decodes character sequences using CTC loss.

**`allowlist='0123456789'`** constrains the CRNN decoder's output vocabulary to digit characters only. Without this:
- The sign border may be detected as a text region ("O" from the circle)
- "mph" text sometimes present below the number would be returned
- CTC may confuse "6" with "G" or "1" with "I" in ambiguous cases

`detail=1` returns `(bounding_box, text, confidence)` tuples. The bounding box is relative to the preprocessed crop (not the original image), so it is not propagated to the output message.

---

## Step 5: Speed Validation

```python
_VALID_SPEEDS = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}

for (_, text, conf) in ocr_results:
    if conf < conf_threshold:
        continue
    digits = ''.join(c for c in text if c.isdigit())
    value = int(digits)
    if value in _VALID_SPEEDS and conf > best_conf:
        best_speed, best_conf = value, conf
```

Validation rejects:
- **Partial reads** — "5" from a partially occluded "55"
- **Year numbers** — sticker artifacts like "2024"
- **County numbers** — e.g. "County Road 130"
- **Reflector IDs** — small numbers embossed on sign hardware

The `_VALID_SPEEDS` set covers all posted speed limits used on US roads from 15 mph (school zones) to 85 mph (the single 85 mph Texas State Highway 130 maximum). Values outside this set are discarded regardless of OCR confidence.

If multiple results are returned (e.g. "45" and "4" both found), the **highest-confidence valid reading** is selected.

---

## Common Failure Modes

| Condition | Symptom | Mitigation |
|-----------|---------|------------|
| Sign too small (<15 px tall) | No result published | Upscaling only helps down to ~20 px; detector must fire first |
| Heavy rain / glare on sign | Low OCR confidence, filtered out | Adjust `ocr_confidence_threshold` downward (tradeoff: more false positives) |
| Sign partially occluded | Partial digit read, rejected by whitelist | No mitigation; partial coverage is indeterminate |
| Sign at steep angle | Character distortion → misread | CRAFT handles mild perspective; severe angles fail |
| Shadow across sign | CLAHE mostly compensates | Adjust `clipLimit` higher for extreme cases |
| EasyOCR not yet init | Node not ready | Reader loads at startup; latency before first callback is normal |

---

## Performance Notes

- EasyOCR runs on GPU when `device: "0"` is set. GPU inference is ~5–10× faster than CPU for CRAFT.
- The `Reader` object must be created **once** at node init. Re-creating it per callback adds ~2 s overhead per frame.
- First run downloads ~100 MB of model files to `~/.EasyOCR/model/`. Subsequent runs use the cached files.
- If only CPU is available, EasyOCR still works but OCR latency increases to ~200–500 ms per crop.
