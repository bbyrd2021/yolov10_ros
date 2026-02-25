# yolov10_ros — Pre-Deployment Sanity Checklist

> **Purpose:** Feed this document to Claude Code before running `catkin_make` or launching on the vehicle.  
> **Usage:** `cd ~/ADC3_ROS/catkin_wsADC/src/yolov10_ros && claude`  
> Then prompt: *"Read PRE_DEPLOY_CHECKS.md and perform every check listed."*

---

## 1. Environment Checks

### 1.1 ROS Installation
```bash
# Must return "noetic"
rosversion -d

# Must resolve without error
which roslaunch rosrun rosmsg
```
**Expected:** ROS Noetic. Any other distro means launch files and message dependencies may not resolve.

### 1.2 Workspace Source
```bash
# Check that the workspace overlay is active
echo $ROS_PACKAGE_PATH | tr ':' '\n' | grep catkin_wsADC
```
**Expected:** The `catkin_wsADC/src` path appears. If not, run:
```bash
source ~/ADC3_ROS/catkin_wsADC/devel/setup.bash
```

### 1.3 Python Version
```bash
python3 --version
which python3
```
**Expected:** Python 3.8.x (ships with Ubuntu 20.04 / ROS Noetic). Nodes use `#!/usr/bin/env python3` — if `python3` resolves to 3.10+ there may be subtle compatibility issues with older torch/torchvision wheels.

---

## 2. Python Dependency Checks

Run the following and verify every package is present. Missing packages will cause silent import errors at node startup.

```bash
python3 -c "import rospy;          print('rospy OK')"
python3 -c "import cv2;            print('cv2 OK:', cv2.__version__)"
python3 -c "import torch;          print('torch OK:', torch.__version__)"
python3 -c "import torchvision;    print('torchvision OK:', torchvision.__version__)"
python3 -c "import ultralytics;    print('ultralytics OK:', ultralytics.__version__)"
python3 -c "import easyocr;        print('easyocr OK')"
python3 -c "import message_filters;print('message_filters OK')"
python3 -c "from cv_bridge import CvBridge; print('cv_bridge OK')"
```

### 2.1 GPU Availability
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```
**Expected:** `CUDA available: True`, device count ≥ 1. If False, all nodes will fall back to CPU — update `device` params to `"cpu"` in all config YAMLs before launching.

### 2.2 Ultralytics YOLOv10 Import
```bash
python3 -c "from ultralytics import YOLOv10; print('YOLOv10 import OK')"
```
If this fails, the detector node falls back to `YOLO` — confirm the fallback path is present in `src/detector_node.py`:
```python
except ImportError:
    from ultralytics import YOLO
```

---

## 3. Model Weight Checks

All three weight files must exist and be non-zero. The nodes will crash at startup if any are missing.

```bash
ls -lh ~/ADC3_ROS/catkin_wsADC/src/yolov10_ros/models/
```

### Required Files

| File | Min Size | Used By |
|------|----------|---------|
| `yolov10-bdd-vanilla.pt` | ~30 MB | `detector_node.py` |
| `efficientnet_b0_20260211_195116_light_cls.pth` | ~15 MB | `light_classifier_node.py` |
| `efficientnet_b0_20260202_154542_sign_cls.pth` | ~15 MB | `sign_classifier_node.py` |

**Check each individually:**
```bash
for f in \
  "yolov10-bdd-vanilla.pt" \
  "efficientnet_b0_20260211_195116_light_cls.pth" \
  "efficientnet_b0_20260202_154542_sign_cls.pth"; do
    path="models/$f"
    [ -f "$path" ] && echo "FOUND: $f" || echo "MISSING: $f"
done
```

### 3.1 Checkpoint Key Validation
The `.pth` files must contain the key `model_state_dict`. Verify before loading:
```bash
python3 -c "
import torch
for f in ['models/efficientnet_b0_20260211_195116_light_cls.pth',
          'models/efficientnet_b0_20260202_154542_sign_cls.pth']:
    ckpt = torch.load(f, map_location='cpu')
    keys = list(ckpt.keys())
    print(f, '->', keys)
    assert 'model_state_dict' in keys, 'MISSING model_state_dict key!'
print('Checkpoint keys OK')
"
```

### 3.2 Classifier Head Shape Validation
```bash
python3 -c "
import torch
from torchvision.models import efficientnet_b0

for path, n_classes in [
    ('models/efficientnet_b0_20260211_195116_light_cls.pth', 6),
    ('models/efficientnet_b0_20260202_154542_sign_cls.pth', 15),
]:
    m = efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_classes)
    ckpt = torch.load(path, map_location='cpu')
    m.load_state_dict(ckpt['model_state_dict'])
    print(f'LOADED OK: {path} ({n_classes} classes)')
"
```

---

## 4. Package Structure Checks

Claude Code should verify the following files exist at these exact paths. Any missing file will cause `catkin_make` or the launch to fail.

```
yolov10_ros/
├── CMakeLists.txt                        ← catkin build definition
├── package.xml                           ← ROS package metadata
├── setup.py                              ← Python package install
├── config/
│   ├── detector_params.yaml
│   ├── light_classifier_params.yaml
│   ├── sign_classifier_params.yaml
│   └── speed_limit_ocr_params.yaml
├── launch/
│   ├── detector.launch
│   ├── pipeline.launch
│   ├── pipeline_d435.launch
│   └── pipeline_kitti.launch
├── models/
│   └── .gitkeep                          ← weights go here (gitignored)
├── msg/
│   ├── Detection.msg
│   ├── DetectionArray.msg
│   ├── ClassifiedLight.msg
│   ├── ClassifiedLightArray.msg
│   ├── ClassifiedSign.msg
│   ├── ClassifiedSignArray.msg
│   ├── SpeedLimit.msg
│   └── SpeedLimitArray.msg
└── src/
    ├── detector_node.py
    ├── light_classifier_node.py
    ├── sign_classifier_node.py
    ├── speed_limit_ocr_node.py
    └── yolov10_ros/
        ├── __init__.py
        ├── image_utils.py
        └── visualization.py
```

**Quick file existence check:**
```bash
python3 -c "
import os
required = [
    'CMakeLists.txt', 'package.xml', 'setup.py',
    'config/detector_params.yaml',
    'config/light_classifier_params.yaml',
    'config/sign_classifier_params.yaml',
    'config/speed_limit_ocr_params.yaml',
    'launch/detector.launch',
    'launch/pipeline.launch',
    'launch/pipeline_d435.launch',
    'launch/pipeline_kitti.launch',
    'msg/Detection.msg', 'msg/DetectionArray.msg',
    'msg/ClassifiedLight.msg', 'msg/ClassifiedLightArray.msg',
    'msg/ClassifiedSign.msg', 'msg/ClassifiedSignArray.msg',
    'msg/SpeedLimit.msg', 'msg/SpeedLimitArray.msg',
    'src/detector_node.py',
    'src/light_classifier_node.py',
    'src/sign_classifier_node.py',
    'src/speed_limit_ocr_node.py',
    'src/yolov10_ros/__init__.py',
    'src/yolov10_ros/image_utils.py',
    'src/yolov10_ros/visualization.py',
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print('MISSING FILES:')
    for f in missing: print(' -', f)
else:
    print('All required files present.')
"
```

---

## 5. CMakeLists.txt Checks

Verify these items manually or with Claude Code:

- [ ] `project(yolov10_ros)` is set
- [ ] `catkin_python_setup()` is called (required for `src/yolov10_ros/` to be importable)
- [ ] All 8 `.msg` files are listed under `add_message_files(FILES ...)`
- [ ] `generate_messages(DEPENDENCIES std_msgs sensor_msgs)` is present
- [ ] All four node scripts are listed under `catkin_install_python(PROGRAMS ...)`

```bash
# Quick grep checks
grep -n "catkin_python_setup"   CMakeLists.txt
grep -n "add_message_files"     CMakeLists.txt
grep -n "generate_messages"     CMakeLists.txt
grep -n "catkin_install_python" CMakeLists.txt
```
**Expected:** Each returns exactly one match.

---

## 6. Message File Checks

Each `.msg` file must define the correct fields. Mismatched field names will cause `AttributeError` at runtime.

### Detection.msg
Must contain: `string class_name`, `float64 confidence`, `int32 xmin ymin xmax ymax`

### ClassifiedLight.msg
Must contain: `string state`, `float64 state_confidence`, `int32 xmin ymin xmax ymax`

### ClassifiedSign.msg
Must contain: `string sign_type`, `float64 type_confidence`, `int32 xmin ymin xmax ymax`

### SpeedLimit.msg
Must contain: `int32 speed`, `string display`, `float64 ocr_confidence`, `int32 xmin ymin xmax ymax`

**After building**, verify compiled messages:
```bash
rosmsg show yolov10_ros/Detection
rosmsg show yolov10_ros/ClassifiedLight
rosmsg show yolov10_ros/ClassifiedSign
rosmsg show yolov10_ros/SpeedLimit
```

---

## 7. Node Script Checks

### 7.1 Shebang Lines
All node scripts must start with `#!/usr/bin/env python3`:
```bash
head -1 src/detector_node.py
head -1 src/light_classifier_node.py
head -1 src/sign_classifier_node.py
head -1 src/speed_limit_ocr_node.py
```

### 7.2 Execute Permissions
```bash
ls -la src/*.py
```
**Expected:** All four files show `-rwxr-xr-x`. If not:
```bash
chmod +x src/detector_node.py src/light_classifier_node.py \
         src/sign_classifier_node.py src/speed_limit_ocr_node.py
```

### 7.3 Import Consistency
Claude Code should verify that field names used in each node match the corresponding `.msg` definitions:

| Node | Accesses These Fields |
|------|-----------------------|
| `detector_node.py` | `Detection.class_name`, `.confidence`, `.xmin/.ymin/.xmax/.ymax` |
| `light_classifier_node.py` | `ClassifiedLight.state`, `.state_confidence`, `.xmin/.ymin/.xmax/.ymax` |
| `sign_classifier_node.py` | `ClassifiedSign.sign_type`, `.type_confidence`, `.xmin/.ymin/.xmax/.ymax` |
| `speed_limit_ocr_node.py` | `SpeedLimit.speed`, `.display`, `.ocr_confidence`, `.xmin/.ymin/.xmax/.ymax` |

### 7.4 Class Counts
Verify the hardcoded class counts match the model checkpoint heads:
```bash
grep -n "num_classes\|len(self.class_map)\|LIGHT_CLASSES\|SIGN_CLASSES" \
  src/light_classifier_node.py src/sign_classifier_node.py
```
**Expected:** Light classifier → 6 classes. Sign classifier → 15 classes.

---

## 8. Launch File Checks

### 8.1 Weight Path Resolution
The launch files use `$(find yolov10_ros)/models/...`. Verify the package is findable:
```bash
rospack find yolov10_ros
```
**Expected:** Returns the absolute path to the package. If it errors, the workspace isn't sourced.

### 8.2 Config File References
Verify the `<rosparam file="...">` paths in `pipeline.launch` point to files that exist:
```bash
grep "rosparam file" launch/pipeline.launch
```
Each referenced `.yaml` must exist under `config/`.

### 8.3 Node Type Names
The `type="..."` attribute in each `<node>` tag must exactly match the filename in `src/`:
```bash
grep 'type=' launch/pipeline.launch
```
Expected values: `detector_node.py`, `light_classifier_node.py`, `sign_classifier_node.py`, `speed_limit_ocr_node.py`

---

## 9. Camera Topic Check

Before launching on the car, confirm the camera is publishing and the topic name matches what the launch file expects.

```bash
# List active image topics
rostopic list | grep -i "image\|camera"

# Check the topic the launch file subscribes to (default)
grep "input_image_topic" config/detector_params.yaml launch/pipeline.launch
```

**Common topic mismatches:**

| Camera | Actual Topic | Launch Arg Needed |
|--------|-------------|------------------|
| RealSense D435 | `/camera/color/image_raw` | *(default — no change)* |
| Ouster + cam | May differ | `input_image_topic:=/your/topic` |
| KITTI bag | `/kitti/camera_color_left/image_raw` | Use `pipeline_kitti.launch` |

---

## 10. Post-Build Verification

Run these after `catkin_make` succeeds:

```bash
# 1. Source the new devel overlay
source ~/ADC3_ROS/catkin_wsADC/devel/setup.bash

# 2. Verify messages were generated
rosmsg list | grep yolov10_ros

# 3. Verify Python package is importable
python3 -c "from yolov10_ros.image_utils import imgmsg_to_cv2; print('image_utils OK')"
python3 -c "from yolov10_ros.visualization import draw_detections; print('visualization OK')"

# 4. Dry-run the detector node (no roscore needed, just checks imports)
python3 -c "
import sys
sys.argv = ['detector_node.py']
# Patch rospy.init_node to no-op
import unittest.mock as mock
with mock.patch('rospy.init_node'), mock.patch('rospy.spin'):
    import importlib.util, os
    spec = importlib.util.spec_from_file_location('detector_node', 'src/detector_node.py')
    mod = importlib.util.load_from_spec(spec)
" 2>&1 | grep -v "^$" || echo "Import check complete"
```

---

## 11. Known Gotchas

| Issue | Symptom | Fix |
|-------|---------|-----|
| `devel/setup.bash` not sourced | `rosmsg show` errors, `rospack find` fails | `source ~/ADC3_ROS/catkin_wsADC/devel/setup.bash` |
| Wrong `catkin_make` target | Other packages in workspace rebuild and fail | Use `catkin_make --only-pkg-with-deps yolov10_ros` |
| `model_state_dict` key missing | `KeyError` at node startup | Weights file is corrupted or wrong format — re-copy from source |
| `ApproximateTimeSynchronizer` never fires | Classifiers publish nothing | Check detector is publishing both `/yolov10/detections` AND `/yolov10/image_raw` |
| EasyOCR first-run download | OCR node hangs for 60+ seconds | Normal — it downloads ~100 MB to `~/.EasyOCR/` on first use |
| `cv_bridge` encoding error | `[ERROR] bgr8 is not a color format` | Source image may be mono; check camera encoding with `rostopic echo /camera/color/image_raw/header` |
| GPU OOM with all nodes on same device | CUDA out of memory | Set `device: "cpu"` for light/sign classifiers in their YAMLs — EfficientNet B0 is lightweight |

---

## 12. Minimal Smoke Test (No Car Required)

Test the full message pipeline using a static image without a camera:

```bash
# Terminal 1 — start roscore
roscore

# Terminal 2 — launch detector only with a test image publisher
roslaunch yolov10_ros detector.launch \
  input_image_topic:=/test/image \
  view_image:=false

# Terminal 3 — publish a single test frame
python3 -c "
import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
rospy.init_node('test_pub')
pub = rospy.Publisher('/test/image', Image, queue_size=1)
bridge = CvBridge()
img = cv2.imread('/path/to/any/test/image.jpg')
rospy.sleep(1)
pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
rospy.sleep(2)
"

# Terminal 4 — verify detections are published
rostopic echo /yolov10/detections --noarr -n 1
```

---

## 13. Dependency Documentation (REQUIRED)

> **Rule: Any package, library, or system dependency added to this project — for any reason — must be logged here before merging or deploying. No exceptions. This applies to Python packages, ROS packages, apt packages, and model files.**
>
> This log is the single source of truth for reproducing the environment on a new machine, re-imaging the car's computer, or onboarding a new team member.

### 13.1 How to Log a New Dependency

When you install anything new, add a row to the appropriate table below. Claude Code should enforce this — if it installs a package as part of a fix or feature, it must update this section before finishing.

**Template:**
```
| package-name | version | why it was added | who added it | date |
```

---

### 13.2 Python Packages (pip3)

| Package | Version | Purpose | Added By | Date |
|---------|---------|---------|----------|------|
| `ultralytics` | 8.4.14 | YOLOv10 model loading and inference (uses generic YOLO fallback; YOLOv10 class removed in 8.x) | bbyrd | 2026-02 |
| `torch` | 2.1.2+cu121 | PyTorch deep learning backend | bbyrd | 2026-02 |
| `torchvision` | 0.16.2+cu121 | EfficientNet B0 model architecture | bbyrd | 2026-02 |
| `opencv-python-headless` | 4.13.0.92 | Image processing, ROI cropping (replaces opencv-python; installed as easyocr dep) | bbyrd | 2026-02 |
| `easyocr` | 1.7.2 | CRAFT + CRNN OCR for speed limit signs | bbyrd | 2026-02 |

> When adding a new pip package, also pin the version:
> ```bash
> pip3 show <package> | grep Version
> ```
> Then add that version to the table above.

---

### 13.3 ROS Packages (apt)

| Package | Version | Purpose | Added By | Date |
|---------|---------|---------|----------|------|
| `ros-noetic-cv-bridge` | system | ROS ↔ OpenCV image conversion | bbyrd | 2026-02 |
| `ros-noetic-image-transport` | system | Compressed image topic support | bbyrd | 2026-02 |
| `ros-noetic-message-filters` | system | ApproximateTimeSynchronizer for classifier nodes | bbyrd | 2026-02 |
| `ros-noetic-rosbag` | system | Bag playback for offline testing | bbyrd | 2026-02 |
| `python3-catkin-tools` | system | `catkin build` command | bbyrd | 2026-02 |

---

### 13.4 System / apt Packages

| Package | Purpose | Added By | Date |
|---------|---------|----------|------|
| `python3-pip` | Python package manager | bbyrd | 2026-02 |

---

### 13.5 Model Files

| File | Architecture | Classes | Trained On | Added By | Date |
|------|-------------|---------|------------|----------|------|
| `yolov10-bdd-vanilla.pt` | YOLOv10 | 10 (BDD100K) | BDD100K driving dataset | bbyrd | 2026-02 |
| `efficientnet_b0_20260211_195116_light_cls.pth` | EfficientNet B0 | 6 (light states) | Cropped traffic light images | bbyrd | 2026-02 |
| `efficientnet_b0_20260202_154542_sign_cls.pth` | EfficientNet B0 | 15 (sign types) | Cropped US traffic sign images | bbyrd | 2026-02 |

---

### 13.6 Checking for Undocumented Packages

Claude Code should run this audit before every deployment to catch anything installed but not yet logged:

```bash
# Dump current pip environment
pip3 freeze > /tmp/current_pip.txt

# Check for packages present in environment but not in this doc
# (manually compare against table 13.2)
grep -E "ultralytics|torch|torchvision|opencv|easyocr" /tmp/current_pip.txt

# Dump installed ROS packages
dpkg -l | grep ros-noetic | awk '{print $2, $3}'
```

If anything appears in the environment that is **not** in the tables above, **stop and document it before proceeding.**

---

### 13.7 Reproducing the Full Environment from Scratch

If the car's computer is re-imaged or a new dev machine is set up, run in order:

```bash
# 1. ROS Noetic (Ubuntu 20.04)
sudo apt install -y \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-message-filters \
  ros-noetic-rosbag \
  python3-catkin-tools \
  python3-pip

# 2. Python packages
pip3 install \
  ultralytics \
  torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
  opencv-python \
  easyocr

# 3. Build the package
cd ~/ADC3_ROS/catkin_wsADC
catkin_make --only-pkg-with-deps yolov10_ros
source devel/setup.bash

# 4. Copy model weights into models/
# (transfer from external drive or shared storage)
```

> **Any deviation from the above sequence must be documented in sections 13.2–13.4 before the next deployment.**

---

*Last updated for car deployment — NC A&T SAE AutoDrive Challenge II*