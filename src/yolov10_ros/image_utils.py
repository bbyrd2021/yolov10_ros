import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

_bridge = CvBridge()


def imgmsg_to_cv2(msg, encoding="bgr8"):
    """Convert a ROS Image or CompressedImage message to a cv2 BGR numpy array."""
    if isinstance(msg, CompressedImage):
        return _bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=encoding)
    return _bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


def cv2_to_imgmsg(image, encoding="bgr8"):
    """Convert a cv2 BGR numpy array to a ROS Image message."""
    return _bridge.cv2_to_imgmsg(image, encoding=encoding)


def crop_roi(image, xmin, ymin, xmax, ymax, padding=0):
    """Crop a bounding box region from an image with optional padding.

    Coordinates are clamped to image bounds.
    Returns the cropped BGR numpy array.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(xmin) - padding)
    y1 = max(0, int(ymin) - padding)
    x2 = min(w, int(xmax) + padding)
    y2 = min(h, int(ymax) + padding)
    return image[y1:y2, x1:x2]
