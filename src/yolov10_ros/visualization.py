import cv2
import numpy as np

# Color palette for different classes
_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (128, 0, 255),
]


def _get_color(class_name):
    """Deterministic color for a given class name."""
    return _COLORS[hash(class_name) % len(_COLORS)]


def draw_detections(image, detections):
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR numpy array
        detections: list of objects with class_name, confidence, xmin, ymin, xmax, ymax

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    for det in detections:
        color = _get_color(det.class_name)
        pt1 = (int(det.xmin), int(det.ymin))
        pt2 = (int(det.xmax), int(det.ymax))
        cv2.rectangle(annotated, pt1, pt2, color, 2)

        label = "{} {:.2f}".format(det.class_name, det.confidence)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle for text
        cv2.rectangle(annotated,
                      (pt1[0], pt1[1] - th - baseline - 4),
                      (pt1[0] + tw, pt1[1]),
                      color, -1)
        cv2.putText(annotated, label,
                    (pt1[0], pt1[1] - baseline - 2),
                    font, font_scale, (0, 0, 0), thickness)

    return annotated
