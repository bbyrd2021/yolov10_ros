"""
yolov10_ros Python package
===========================
Shared utilities for the yolov10_ros ROS nodes.

Modules
-------
image_utils
    ROS ↔ OpenCV conversion helpers (imgmsg_to_cv2, cv2_to_imgmsg, crop_roi).
    Transparent handling of both sensor_msgs/Image and CompressedImage.

visualization
    OpenCV annotation helpers (draw_detections).  Used by the detector node
    for optional annotated-image publishing and by the standalone demo scripts.

Note
----
The generated ROS message classes (DetectionArray, ClassifiedLightArray, etc.)
are imported via ``from yolov10_ros.msg import ...`` and are NOT part of this
Python module — they live in the catkin-generated ``devel/`` overlay.
"""
