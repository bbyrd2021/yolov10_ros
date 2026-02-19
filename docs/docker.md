# Docker Guide

ROS 1 Noetic requires Ubuntu 20.04 (Focal). On an Ubuntu 22.04 (Jammy) development machine, the cleanest solution is to run the full pipeline inside a Docker container based on the official `ros:noetic-robot` image.

---

## Prerequisites

### 1. Docker Engine

```bash
# Verify Docker is installed
docker --version   # should be 20.10+
```

If not installed: [docs.docker.com/engine/install/ubuntu](https://docs.docker.com/engine/install/ubuntu/)

### 2. NVIDIA Container Toolkit

Required for GPU passthrough (`runtime: nvidia` in docker-compose).

```bash
# Check if already installed
nvidia-smi   # should show GPU info from inside a container

# Install if missing
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
     | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Model weights

Place model files in the repo-level `models/` directory before building:

```
AutoDrivePerception2026/
└── models/
    ├── yolov10-bdd-vanilla.pt
    ├── efficientnet_b0_20260211_195116_light_cls.pth
    └── efficientnet_b0_20260202_154542_sign_cls.pth
```

The `yolov10_ros/models/` path is a symlink to this directory.

---

## Directory Layout

The `Dockerfile` and `docker-compose.yml` live inside `yolov10_ros/`. Run all Docker commands from that directory.

```
yolov10_ros/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── data/              ← create this; place kitti.bag here
```

---

## Build

```bash
cd /path/to/AutoDrivePerception2026/yolov10_ros
docker compose build
```

The build:
1. Pulls `ros:noetic-robot` (Ubuntu 20.04 + ROS Noetic)
2. Installs system packages (`python3-catkin-tools`, `ros-noetic-cv-bridge`, etc.)
3. Installs Python packages (`ultralytics`, `torch`, `torchvision`, `opencv-python-headless`, `easyocr`)
4. Creates the catkin workspace skeleton at `/catkin_ws`
5. Copies `entrypoint.sh`

**Note:** PyTorch is installed with CUDA 11.8 wheels. If your host GPU requires a different CUDA version, change the `--index-url` in the Dockerfile:
```dockerfile
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Available index URLs: `cu118`, `cu121`, `cu124`, `cpu`.

---

## Running the Perception Pipeline

```bash
docker compose up
```

What happens:
1. Container starts
2. `entrypoint.sh` runs: sources ROS → `catkin build yolov10_ros` → sources devel overlay
3. `roslaunch yolov10_ros pipeline_kitti.launch` starts all four nodes

The pipeline sits idle waiting for images on `/kitti/camera_color_left/image_raw`.

---

## Playing a KITTI Bag

In a second terminal, start the bag player service:

```bash
docker compose run bagplayer
```

This replays `./data/kitti.bag` at 0.5× speed with simulated clock. Slow playback gives the OCR node time to process each frame.

**Adjusting playback speed:**

Edit the `command` in `docker-compose.yml`:
```yaml
rosbag play /data/kitti.bag --clock -r 1.0   # full speed
rosbag play /data/kitti.bag --clock -r 0.25  # quarter speed
```

**Converting KITTI raw sequences to a bag:**

If you have KITTI raw data (folders of images), use [kitti2bag](https://github.com/tomas789/kitti2bag):
```bash
pip install kitti2bag
kitti2bag -t 2011_09_26 -r 0001 raw_synced .
```
This creates `kitti_2011_09_26_drive_0001_synced.bag` with standard topic names.

---

## Monitoring Topics

```bash
# Open a shell in the running perception container
docker exec -it $(docker ps -qf "name=perception") bash

# Inside the container:
source /catkin_ws/devel/setup.bash
rostopic list
rostopic hz /yolov10/detections
rostopic echo /perception/speed_limit
```

---

## Viewing Annotated Output (Optional)

For X11 forwarding (rviz, cv2.imshow):

```bash
# On the host, allow connections from Docker
xhost +local:docker

# Enable view_image in detector_params.yaml
view_image: true
```

The `/tmp/.X11-unix` socket is already mounted in `docker-compose.yml`.

---

## Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./yolov10_ros` | `/catkin_ws/src/yolov10_ros` | Package source (live edits) |
| `./models` | `/catkin_ws/src/yolov10_ros/models` | Model weights |
| `./data` | `/data` | KITTI bags, output videos |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 display forwarding |

The package source is live-mounted, meaning Python edits on the host take effect on the **next container startup** (the entrypoint re-runs `catkin build` every time). You do not need to rebuild the image for code changes.

---

## Common Issues

### `runtime: nvidia` not found

```
ERROR: Service 'perception' failed to build: runtime 'nvidia' not found
```

Install the NVIDIA Container Toolkit (see Prerequisites above) and restart Docker.

---

### EasyOCR downloading on every container start

EasyOCR downloads models to `~/.EasyOCR/` inside the container. Add a volume mount to persist the cache:

```yaml
volumes:
  - ~/.EasyOCR:/root/.EasyOCR   # add to perception service
```

---

### `catkin build` fails: missing package

```
Could not find a package configuration file provided by "cv_bridge"
```

Rebuild the image from scratch (the base apt cache may be stale):

```bash
docker compose build --no-cache
```

---

### Topics not visible from host

Verify `network_mode: host` is set in `docker-compose.yml`. Without host networking, the container's ROS master is isolated from the host.
