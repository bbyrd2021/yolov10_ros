#!/bin/bash
# entrypoint.sh
# -------------
# Docker container entrypoint for the yolov10_ros perception stack.
#
# Executed automatically on every `docker run` / `docker compose up`.
# Sources ROS, builds the package (so host-side Python edits are picked up
# without a full image rebuild), sources the devel overlay, then execs
# whatever command was passed to the container.
#
# The `set -e` flag causes the script to abort immediately on any error,
# preventing a silent failure from leaving the container running in a broken
# state (e.g. if catkin build fails due to a syntax error in a node).

set -e

# ── Source ROS Noetic base setup ──────────────────────────────────────────────
# Makes roslaunch, rostopic, catkin, etc. available in PATH.
source /opt/ros/noetic/setup.bash

# ── Build yolov10_ros ──────────────────────────────────────────────────────────
# --no-status suppresses the catkin build progress bar (cleaner Docker logs).
# Only the yolov10_ros package is built to keep startup time short; the
# workspace may contain other packages (e.g. from ros:noetic-robot) that are
# already installed system-wide and don't need rebuilding.
echo "Building yolov10_ros..."
cd /catkin_ws
catkin build yolov10_ros --no-status

# ── Source devel overlay ──────────────────────────────────────────────────────
# Makes the generated message types (DetectionArray, ClassifiedLightArray, etc.)
# importable by the Python nodes.  Must be sourced AFTER the build completes.
source /catkin_ws/devel/setup.bash

# ── Execute the requested command ─────────────────────────────────────────────
# `exec "$@"` replaces the shell process with the command, so signals (SIGTERM,
# SIGINT) are delivered directly to the process rather than to the shell.
# This ensures `docker stop` and Ctrl-C work correctly.
echo "Build complete. Running: $@"
exec "$@"
