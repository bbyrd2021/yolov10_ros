# Pipeline Architecture

## Overview

The perception stack is a linear ROS 1 graph of four nodes, each responsible for one stage of understanding. The design principle is **separation of concerns**: the detector doesn't know about sign types, the sign classifier doesn't know about speed numbers, and the OCR node doesn't know about anything except the crop it receives. Every node publishes a structured message that can be consumed independently by the autonomy stack or by other tools (rviz, rosbag, rqt).

```
 Camera
   │  /camera/color/image_raw
   │  (sensor_msgs/Image or CompressedImage)
   ▼
┌─────────────────────────────────────┐
│         yolov10_detector            │  YOLOv10, BDD100K weights
│                                     │  640×640 input, NMS-free
└──────────────┬──────────────────────┘
               │
         ┌─────┴──────────────────────────────┐
         │                                    │
  /yolov10/detections                 /yolov10/image_raw
  (DetectionArray)                    (Image — decoded BGR)
         │                                    │
    ┌────┴────┐                          ┌────┴────┐
    │         │                          │         │
    ▼         ▼                          ▼         ▼
┌────────┐ ┌────────┐           ┌────────────────────────┐
│ light  │ │  sign  │           │  ApproximateTimeSynchronizer
│classif.│ │classif.│           │  pairs DetectionArray + Image
└───┬────┘ └───┬────┘           └────────────────────────┘
    │           │
    │           │  /perception/traffic_signs (ClassifiedSignArray)
    │           │
    │      ┌────┴──────────────────┐
    │      │   speed_limit_ocr     │  EasyOCR
    │      │   (only speed_limit   │  CRAFT + CRNN
    │      │    signs)             │
    │      └────┬──────────────────┘
    │           │
    ▼           ▼
/perception/   /perception/speed_limit
traffic_lights (SpeedLimitArray)
(ClassifiedLightArray)
```

## Node Responsibilities

### 1. `yolov10_detector`

**What it does:** Converts incoming camera frames to BGR arrays, runs the YOLOv10 model, and publishes bounding boxes for every object above the confidence threshold.

**Key design choices:**
- Subscribes to **either** `sensor_msgs/Image` or `sensor_msgs/CompressedImage` — the topic type is queried from rosmaster at startup so no manual configuration is needed.
- Republishes the decoded source frame on `/yolov10/image_raw`. This is intentional: downstream classifier nodes need to crop ROIs from the exact same pixel data that produced the bounding boxes. Republishing once avoids each classifier node needing its own camera subscription.
- Uses `queue_size=1` with `buff_size=2**24` on the image subscriber to always process the most recent frame and never build a backlog when inference is slower than the camera rate.
- YOLOv10 is **NMS-free** (end-to-end trained with a one-to-one assignment head), so there is no post-processing NMS step.

**It does NOT:** know about traffic light states, sign types, or speed values.

---

### 2. `light_classifier_node`

**What it does:** For every frame, filters `DetectionArray` for boxes with `class_name == "traffic light"`, crops each ROI (with padding), runs an EfficientNet B0 model to predict the light state, and publishes `ClassifiedLightArray`.

**Key design choices:**
- Uses `message_filters.ApproximateTimeSynchronizer` to pair `DetectionArray` with `Image` by header timestamp (slop=0.1 s). This guarantees that crops are taken from the frame that actually produced those bounding boxes — important if inference latency causes frames to arrive slightly out of sync.
- Only publishes when at least one light passes the confidence gate, which avoids flooding downstream consumers with empty messages.
- The EfficientNet preprocessing (resize → RGB → normalize) must match training exactly. ImageNet mean/std normalisation is hardcoded.

**It does NOT:** re-run detection or process non-light detections.

---

### 3. `sign_classifier_node`

**What it does:** Mirror of the light classifier, but operating on `class_name == "traffic sign"` detections with a 15-class EfficientNet B0 model.

**Key design choices:**
- `speed_limit` signs are included in the published `ClassifiedSignArray` without any special treatment. The sign classifier's job is to identify *what type* of sign is present; the downstream OCR node handles *what value* is on it.
- The 15-class model covers both regulatory (stop, yield, no_left_turn) and warning (roadwork, railroad_crossing) categories, giving the autonomy stack everything it needs from a single subscription.

**It does NOT:** run OCR, know about speed values, or do anything special with speed limit signs beyond labelling them.

---

### 4. `speed_limit_ocr_node`

**What it does:** Subscribes to `ClassifiedSignArray` and `Image`, filters for `sign_type == "speed_limit"`, preprocesses the crop for OCR (upscale + grayscale + CLAHE), runs EasyOCR with a digit-only allowlist, validates the result against the set of legal US speed limits, and publishes `SpeedLimitArray`.

**Key design choices:**
- EasyOCR's `Reader` is initialized **once in `__init__`**, not per-callback. Loading CRAFT + CRNN weights takes ~2 s; doing this per frame would make the node unusable.
- The digit-only `allowlist='0123456789'` passed to `readtext()` constrains the CRNN decoder to only consider digit characters, which eliminates misreads of the circular sign border, state name text, or other surrounding characters.
- Results are validated against a hardcoded set of valid US posted speeds. This rejects partial reads (e.g. "5" from "55"), year numbers on stickers, and other numeric noise.

**It does NOT:** run object detection, run sign type classification, or process anything other than signs already identified as `speed_limit`.

---

## Time Synchronisation

All classifier and OCR nodes use `message_filters.ApproximateTimeSynchronizer`:

```python
sync = message_filters.ApproximateTimeSynchronizer(
    [det_sub, img_sub], queue_size=10, slop=0.1
)
```

- `slop=0.1` means messages are paired if their header timestamps differ by less than 100 ms.
- The detector copies the camera message header directly to the `DetectionArray` header, so the timestamps should always be identical (not just close). The 100 ms tolerance exists as a safety margin for pipeline jitter.
- `queue_size=10` buffers up to 10 unmatched messages on each side before dropping.

## Topic Graph Summary

| Topic | Type | Producer | Consumer(s) |
|-------|------|----------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Camera | `yolov10_detector` |
| `/yolov10/detections` | `DetectionArray` | detector | light classif., sign classif. |
| `/yolov10/image_raw` | `sensor_msgs/Image` | detector | light classif., sign classif., OCR |
| `/yolov10/annotated` | `sensor_msgs/Image` | detector (optional) | rviz, rqt |
| `/perception/traffic_lights` | `ClassifiedLightArray` | light classif. | autonomy stack |
| `/perception/traffic_signs` | `ClassifiedSignArray` | sign classif. | OCR node, autonomy stack |
| `/perception/speed_limit` | `SpeedLimitArray` | OCR node | autonomy stack |

## Latency Profile

Each stage adds latency. Approximate values on an RTX 3090 at 1080p input:

| Stage | Latency |
|-------|---------|
| Camera → detector (YOLOv10 @ 640) | ~15 ms |
| Detector → light/sign classifiers (EfficientNet B0) | ~5 ms each |
| Sign classifier → OCR (EasyOCR, per crop) | ~50–200 ms |
| End-to-end (no speed limit in frame) | ~25 ms |
| End-to-end (speed limit present) | ~80–250 ms |

OCR is the bottleneck. Because it runs in its own node, it does not block the detector or classifiers — the autonomy stack receives light and sign classifications at full frame rate even when OCR is lagging.
