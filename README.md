# yolov10_ros

ROS 1 (Noetic) perception pipeline for autonomous driving. Runs a YOLOv10 object detector trained on BDD100K, then classifies detected traffic lights and signs using dedicated EfficientNet B0 classifiers.

## Architecture

```
/camera/color/image_raw
         |
         v
+--------------------------+
|   yolov10_detector_node  |
|   (YOLOv10 BDD100K)     |
+-----------+--------------+
            |
            | /yolov10/detections  (DetectionArray)
            | /yolov10/image_raw   (Image)
            |
      +-----+------+
      v            v
+----------+ +----------+
|  light   | |  sign    |
| classif. | | classif. |
|  node    | |  node    |
+----+-----+ +----+-----+
     |            |
     v            v
/perception/   /perception/
traffic_lights traffic_signs
```

The detector publishes bounding boxes and the source frame. Each classifier subscribes to both via a time-synchronized callback, filters for its target class, crops the ROI, runs classification, and publishes enriched results.

## Models

| Model | Architecture | Classes | File |
|-------|-------------|---------|------|
| Detector | YOLOv10 | 10 (BDD100K) | `yolov10-bdd-vanilla.pt` |
| Light classifier | EfficientNet B0 | 6 | `efficientnet_b0_*_light_cls.pth` |
| Sign classifier | EfficientNet B0 | 15 | `efficientnet_b0_*_sign_cls.pth` |

**Detector classes:** pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign

**Light classifier classes:** green_left, green_light, red_left, red_light, yellow_left, yellow_light

**Sign classifier classes:** detour, do_not_enter, go_straight_only, no_left_turn, no_right_turn, no_straight, no_u_turn, pedestrian_crossing, railroad_crossing, roadwork, speed_limit, stop, turn_left_only, turn_right_only, yield

Place weight files in the `models/` directory (gitignored).

## Dependencies

```bash
# Python
pip install ultralytics torch torchvision opencv-python

# ROS (should already be on the vehicle)
# rospy, cv_bridge, sensor_msgs, std_msgs, message_filters, image_transport
```

## Build

```bash
# Symlink or copy into your catkin workspace
ln -s /path/to/yolov10_ros ~/catkin_ws/src/yolov10_ros

# Build
cd ~/catkin_ws
catkin build yolov10_ros
source devel/setup.bash
```

## Usage

**Detector only** (for testing):
```bash
roslaunch yolov10_ros detector.launch
```

**Full pipeline** (detector + both classifiers):
```bash
roslaunch yolov10_ros pipeline.launch
```

**Full pipeline with RealSense D435**:
```bash
roslaunch yolov10_ros pipeline_d435.launch
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `device` | `0` | CUDA device index or `cpu` |
| `input_image_topic` | `/camera/color/image_raw` | Camera topic |
| `detector_weights` | `models/yolov10-bdd-vanilla.pt` | Detector weights path |
| `detector_conf` | `0.5` | Detector confidence threshold |
| `light_weights` | `models/efficientnet_b0_*_light_cls.pth` | Light classifier weights |
| `light_conf` | `0.5` | Light classifier confidence threshold |
| `sign_weights` | `models/efficientnet_b0_*_sign_cls.pth` | Sign classifier weights |
| `sign_conf` | `0.5` | Sign classifier confidence threshold |

## Topics

### Published

| Topic | Type | Source |
|-------|------|--------|
| `/yolov10/detections` | `yolov10_ros/DetectionArray` | Detector |
| `/yolov10/image_raw` | `sensor_msgs/Image` | Detector |
| `/yolov10/annotated` | `sensor_msgs/Image` | Detector (optional) |
| `/perception/traffic_lights` | `yolov10_ros/ClassifiedLightArray` | Light classifier |
| `/perception/traffic_signs` | `yolov10_ros/ClassifiedSignArray` | Sign classifier |

### Subscribed

| Topic | Type | Consumer |
|-------|------|----------|
| `/camera/color/image_raw` | `sensor_msgs/Image` or `CompressedImage` | Detector |
| `/yolov10/detections` | `yolov10_ros/DetectionArray` | Both classifiers |
| `/yolov10/image_raw` | `sensor_msgs/Image` | Both classifiers |

## Custom Messages

**Detection.msg** — single bounding box from the detector
```
string  class_name
float64 confidence
int32   xmin, ymin, xmax, ymax
```

**ClassifiedLight.msg** — traffic light with classified state
```
string  state              # green_light, red_left, yellow_light, etc.
float64 state_confidence
int32   xmin, ymin, xmax, ymax
```

**ClassifiedSign.msg** — traffic sign with classified type
```
string  sign_type          # stop, yield, speed_limit, etc.
float64 type_confidence
int32   xmin, ymin, xmax, ymax
```

Each has an `*Array.msg` wrapper with a `Header` for timestamping.

## Package Structure

```
yolov10_ros/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── config/
│   ├── detector_params.yaml
│   ├── light_classifier_params.yaml
│   └── sign_classifier_params.yaml
├── launch/
│   ├── detector.launch
│   ├── pipeline.launch
│   └── pipeline_d435.launch
├── models/                        # weights go here (gitignored)
├── msg/
│   ├── Detection.msg
│   ├── DetectionArray.msg
│   ├── ClassifiedLight.msg
│   ├── ClassifiedLightArray.msg
│   ├── ClassifiedSign.msg
│   └── ClassifiedSignArray.msg
└── src/
    ├── detector_node.py
    ├── light_classifier_node.py
    ├── sign_classifier_node.py
    └── yolov10_ros/
        ├── __init__.py
        ├── image_utils.py
        └── visualization.py
```

## Notes

- **GPU sharing**: All three models default to the same GPU. Classifiers are lightweight and can be moved to CPU (`device: "cpu"`) if GPU memory is tight.
- **NMS-free**: YOLOv10 is end-to-end — no NMS post-processing step.
- **Frame drops are OK**: All image subscribers use `queue_size=1` to always process the latest frame.
- **Topic sync**: Classifier nodes use `ApproximateTimeSynchronizer` (slop=0.1s) to match detections with the correct source frame.
- **Auto image type detection**: The detector auto-detects `Image` vs `CompressedImage` on the input topic.
