# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — AutoDrive Perception Stack (ROS Noetic)
# ──────────────────────────────────────────────────────────────────────────────
# Provides a self-contained ROS 1 Noetic environment for the yolov10_ros
# package on Ubuntu 22.04 development machines, where Noetic cannot be
# installed natively (it requires Ubuntu 20.04 Focal).
#
# Base image: ros:noetic-robot
#   • Ubuntu 20.04 Focal
#   • ROS Noetic core + robot meta-package (includes rospy, tf, etc.)
#   • No desktop GUI tools — keeps the image small
#
# Build context: yolov10_ros/ directory (run from repo root)
#   docker build -t autodrive-perception .
#
# Typical usage: see docker-compose.yml which handles volume mounts,
# GPU passthrough, and network configuration automatically.
# ──────────────────────────────────────────────────────────────────────────────

FROM ros:noetic-robot

# Suppress interactive apt prompts (e.g. tzdata region selection).
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
# Install ROS packages and build tools in a single layer to minimise image size.
#
#   python3-catkin-tools   — `catkin build` command (preferred over catkin_make)
#   ros-noetic-cv-bridge   — ROS ↔ OpenCV image conversion (used by all nodes)
#   ros-noetic-image-transport — compressed image transport support
#   ros-noetic-message-filters — ApproximateTimeSynchronizer for classifier nodes
#   ros-noetic-rosbag      — Needed by the bagplayer service in docker-compose
#
# `rm -rf /var/lib/apt/lists/*` removes the apt cache to keep the layer small.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install deep learning and vision libraries.
#
#   ultralytics          — YOLOv10 / YOLO model loading and inference
#   torch / torchvision  — PyTorch with CUDA 11.8 binaries (cu118 wheel index)
#                          Change --index-url if your CUDA version differs
#   opencv-python-headless — OpenCV without GUI libs (no display required)
#   easyocr              — CRAFT text detector + CRNN recogniser for OCR node
#
# --no-cache-dir: avoids storing pip wheel cache in the image layer.
RUN pip3 install --no-cache-dir \
    ultralytics \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    opencv-python-headless \
    easyocr

# ── Catkin workspace ──────────────────────────────────────────────────────────
# The package source is NOT copied into the image here — it is mounted at
# runtime by docker-compose so edits on the host take effect without rebuilding.
# Only the workspace skeleton is created.
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Initialise catkin workspace (creates .catkin_tools/ config directory).
# Must source the ROS setup before calling catkin.
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin init"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# The entrypoint script sources ROS, runs `catkin build yolov10_ros`, sources
# the devel overlay, and then execs whatever CMD / command is passed.
# This means every `docker run` automatically rebuilds the package so host-side
# edits to Python nodes are picked up without a full image rebuild.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
