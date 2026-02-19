"""
image_utils.py — ROS ↔ OpenCV conversion helpers
==================================================
Thin wrappers around cv_bridge that add:
  - Transparent handling of both Image and CompressedImage messages
    (the detector node subscribes to whichever type the camera publishes).
  - A bounded ``crop_roi`` helper used by all three classifier nodes.

All images are handled in BGR format (OpenCV native) throughout this package.
"""

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

# Module-level bridge instance — CvBridge is thread-safe and reusing one
# object avoids re-allocating the internal buffer on every frame.
_bridge = CvBridge()


def imgmsg_to_cv2(msg, encoding="bgr8"):
    """Convert a ROS Image or CompressedImage message to a BGR numpy array.

    Automatically dispatches to the correct cv_bridge method based on the
    message type, so callers do not need to check the topic type themselves.

    Args:
        msg:      sensor_msgs/Image or sensor_msgs/CompressedImage.
        encoding: Desired output encoding.  Default "bgr8" (OpenCV native).

    Returns:
        numpy.ndarray of shape (H, W, 3), dtype uint8, in BGR channel order.
    """
    if isinstance(msg, CompressedImage):
        return _bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=encoding)
    return _bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


def cv2_to_imgmsg(image, encoding="bgr8"):
    """Convert a BGR numpy array to a sensor_msgs/Image message.

    Args:
        image:    numpy.ndarray (H, W, 3) uint8 BGR.
        encoding: ROS image encoding string.  Default "bgr8".

    Returns:
        sensor_msgs/Image (header is not populated — callers should copy the
        original message header for timestamp preservation).
    """
    return _bridge.cv2_to_imgmsg(image, encoding=encoding)


def crop_roi(image, xmin, ymin, xmax, ymax, padding=0):
    """Crop a bounding-box region from an image with optional symmetric padding.

    All coordinates are clamped to the image bounds so the function is safe to
    call with boxes that extend to or beyond the image edges (common with
    detections near frame borders).

    Args:
        image:   numpy.ndarray (H, W, 3) BGR source image.
        xmin:    Left edge of the bounding box (pixels, can be float).
        ymin:    Top edge of the bounding box (pixels, can be float).
        xmax:    Right edge of the bounding box (pixels, can be float).
        ymax:    Bottom edge of the bounding box (pixels, can be float).
        padding: Extra pixels to add symmetrically on all four sides.
                 Useful to capture context around tight bounding boxes
                 (e.g. the full traffic-light housing).  Default: 0.

    Returns:
        numpy.ndarray (h, w, 3) BGR crop.  May be empty (size == 0) if the
        input box is degenerate — callers should guard against this.
    """
    h, w = image.shape[:2]
    # Cast to int and clamp to valid pixel range.
    x1 = max(0, int(xmin) - padding)
    y1 = max(0, int(ymin) - padding)
    x2 = min(w, int(xmax) + padding)
    y2 = min(h, int(ymax) + padding)
    return image[y1:y2, x1:x2]
