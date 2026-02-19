# ROS API Reference

Complete reference for all topics, message types, and per-node ROS parameters.

---

## Topics

### Published by `yolov10_detector`

#### `/yolov10/detections` — `yolov10_ros/DetectionArray`

Published every frame. Contains all bounding boxes that passed the confidence threshold.

```
Header      header      # Timestamp from the source camera message
Detection[] detections  # Empty array if nothing detected
```

Each `Detection`:
```
string  class_name    # e.g. "car", "traffic light", "traffic sign"
float64 confidence    # Detector confidence [0.0, 1.0]
int32   xmin          # Left edge (pixels, original image frame)
int32   ymin          # Top edge (pixels, original image frame)
int32   xmax          # Right edge (pixels, original image frame)
int32   ymax          # Bottom edge (pixels, original image frame)
```

#### `/yolov10/image_raw` — `sensor_msgs/Image`

The decoded source frame (BGR, `bgr8` encoding) republished so classifier nodes can crop ROIs without their own camera subscription. Header timestamp is copied from the camera message.

#### `/yolov10/annotated` — `sensor_msgs/Image`

*Optional.* Only published when `publish_annotated: true` in detector params. Bounding boxes with class labels drawn in OpenCV. Useful for rviz or rqt_image_view during development.

---

### Published by `light_classifier_node`

#### `/perception/traffic_lights` — `yolov10_ros/ClassifiedLightArray`

Published only on frames where at least one traffic light passes the confidence threshold.

```
Header            header  # Timestamp from the source DetectionArray
ClassifiedLight[] lights
```

Each `ClassifiedLight`:
```
string  state             # "green_light", "green_left", "red_light",
                          # "red_left", "yellow_light", "yellow_left"
float64 state_confidence  # EfficientNet softmax max [0.0, 1.0]
int32   xmin
int32   ymin
int32   xmax
int32   ymax
```

---

### Published by `sign_classifier_node`

#### `/perception/traffic_signs` — `yolov10_ros/ClassifiedSignArray`

Published only on frames where at least one sign passes the confidence threshold.

```
Header           header  # Timestamp from the source DetectionArray
ClassifiedSign[] signs
```

Each `ClassifiedSign`:
```
string  sign_type         # One of the 15 sign categories (see models.md)
float64 type_confidence   # EfficientNet softmax max [0.0, 1.0]
int32   xmin
int32   ymin
int32   xmax
int32   ymax
```

---

### Published by `speed_limit_ocr_node`

#### `/perception/speed_limit` — `yolov10_ros/SpeedLimitArray`

Published only when at least one speed limit sign was successfully parsed.

```
Header       header
SpeedLimit[] speed_limits
```

Each `SpeedLimit`:
```
int32   speed           # Integer mph value, e.g. 35
string  display         # Human-readable string, e.g. "speed limit: 35"
float64 ocr_confidence  # EasyOCR per-word confidence [0.0, 1.0]
int32   xmin
int32   ymin
int32   xmax
int32   ymax
```

---

## Per-Node ROS Parameters

Parameters are set via `<rosparam file="...config/...yaml"/>` and `<param>` tags in the launch files. All parameters use the `~` (private) namespace.

### `yolov10_detector`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~weights` | str | — | Path to `.pt` weights file |
| `~confidence_threshold` | float | `0.5` | Minimum detection confidence |
| `~inference_size` | int | `640` | Model input resolution (square) |
| `~device` | str | `"0"` | CUDA index or `"cpu"` |
| `~half` | bool | `false` | FP16 inference |
| `~input_image_topic` | str | `/camera/color/image_raw` | Camera subscription topic |
| `~view_image` | bool | `false` | Show live cv2.imshow window |
| `~publish_annotated` | bool | `false` | Publish annotated image topic |
| `~output_topic` | str | `/yolov10/detections` | DetectionArray topic |
| `~output_image_topic` | str | `/yolov10/image_raw` | Source image republish topic |
| `~annotated_image_topic` | str | `/yolov10/annotated` | Annotated image topic |

### `light_classifier_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~weights` | str | — | Path to `.pth` checkpoint |
| `~confidence_threshold` | float | `0.5` | Minimum classifier confidence |
| `~device` | str | `"0"` | CUDA index or `"cpu"` |
| `~target_class` | str | `"traffic light"` | Detector class name to filter for |
| `~roi_padding` | int | `5` | Pixels added around each box before cropping |
| `~class_names` | list | (built-in 6-class map) | Override class list (alphabetical order) |
| `~detection_topic` | str | `/yolov10/detections` | DetectionArray subscription |
| `~image_topic` | str | `/yolov10/image_raw` | Image subscription |
| `~output_topic` | str | `/perception/traffic_lights` | Output topic |

### `sign_classifier_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~weights` | str | — | Path to `.pth` checkpoint |
| `~confidence_threshold` | float | `0.5` | Minimum classifier confidence |
| `~device` | str | `"0"` | CUDA index or `"cpu"` |
| `~target_class` | str | `"traffic sign"` | Detector class name to filter for |
| `~roi_padding` | int | `5` | Pixels added around each box before cropping |
| `~class_names` | list | (built-in 15-class map) | Override class list |
| `~detection_topic` | str | `/yolov10/detections` | DetectionArray subscription |
| `~image_topic` | str | `/yolov10/image_raw` | Image subscription |
| `~output_topic` | str | `/perception/traffic_signs` | Output topic |

### `speed_limit_ocr_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~device` | str | `"0"` | CUDA index for EasyOCR (`"0"` = GPU, `"cpu"` = CPU) |
| `~roi_padding` | int | `10` | Extra padding around sign crop |
| `~min_crop_height` | int | `64` | Minimum crop height before OCR; smaller crops upscaled |
| `~ocr_confidence_threshold` | float | `0.5` | Minimum EasyOCR word confidence |
| `~sign_topic` | str | `/perception/traffic_signs` | ClassifiedSignArray subscription |
| `~image_topic` | str | `/yolov10/image_raw` | Image subscription |
| `~output_topic` | str | `/perception/speed_limit` | SpeedLimitArray output |

---

## Launch File Arguments

### `pipeline.launch`

Used for all full-pipeline deployments (car, Docker, desktop).

| Argument | Default | Description |
|----------|---------|-------------|
| `device` | `0` | GPU index passed to all nodes |
| `input_image_topic` | `/camera/color/image_raw` | Camera topic |
| `detector_weights` | `$(find yolov10_ros)/models/yolov10-bdd-vanilla.pt` | |
| `detector_conf` | `0.5` | Detector confidence threshold |
| `light_weights` | `$(find yolov10_ros)/models/efficientnet_b0_..._light_cls.pth` | |
| `light_conf` | `0.5` | Light classifier threshold |
| `sign_weights` | `$(find yolov10_ros)/models/efficientnet_b0_..._sign_cls.pth` | |
| `sign_conf` | `0.5` | Sign classifier threshold |

Override at launch time:
```bash
roslaunch yolov10_ros pipeline.launch device:=cpu detector_conf:=0.3
```

### `pipeline_kitti.launch`

Thin wrapper around `pipeline.launch` for KITTI bag playback.

| Argument | Default | Description |
|----------|---------|-------------|
| `bag_image_topic` | `/kitti/camera_color_left/image_raw` | KITTI bag image topic |
| `device` | `0` | GPU index |

### `pipeline_d435.launch`

Points the pipeline at a RealSense D435 camera topic.

### `detector.launch`

Standalone detector without classifiers. Useful for verifying detector output before integrating the full pipeline.

---

## Useful `rostopic` Commands

```bash
# Verify detector is running
rostopic hz /yolov10/detections

# Print detected classes
rostopic echo /yolov10/detections | grep class_name

# Watch traffic light states
rostopic echo /perception/traffic_lights

# Watch speed limit readings
rostopic echo /perception/speed_limit

# Check message field layout
rosmsg show yolov10_ros/SpeedLimit
rosmsg show yolov10_ros/ClassifiedSignArray
```
