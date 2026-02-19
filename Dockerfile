FROM ros:noetic-robot

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir \
    ultralytics \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    opencv-python-headless \
    easyocr

# Create catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Bootstrap catkin
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin init"

# The yolov10_ros package is mounted at runtime via docker-compose volume.
# Build happens at container startup so changes are picked up each run.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
