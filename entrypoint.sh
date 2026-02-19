#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash

echo "Building yolov10_ros..."
cd /catkin_ws
catkin build yolov10_ros --no-status

source /catkin_ws/devel/setup.bash

echo "Build complete. Running: $@"
exec "$@"
