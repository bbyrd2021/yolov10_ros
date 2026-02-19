"""
visualization.py — OpenCV annotation helpers
=============================================
Utility functions for drawing bounding boxes and confidence labels directly
onto BGR numpy arrays.  Used by the detector node for optional annotated-image
publishing and for the demo/app scripts.

All drawing is done on a *copy* of the input so the original frame is never
mutated; this is important when multiple nodes need the same source image.
"""

import cv2
import numpy as np

# ── Color palette (BGR) ───────────────────────────────────────────────────────
# A fixed pool of visually distinct colors.  Class names are hashed to indices
# so each class always gets the same color across frames.
_COLORS = [
    (0,   255,   0),    # bright green
    (255,   0,   0),    # blue
    (0,     0, 255),    # red
    (255, 255,   0),    # cyan
    (0,   255, 255),    # yellow
    (255,   0, 255),    # magenta
    (128, 255,   0),    # lime-green
    (255, 128,   0),    # sky-blue
    (0,   128, 255),    # orange
    (128,   0, 255),    # purple
]


def _get_color(class_name):
    """Return a deterministic BGR color for a given class name.

    Uses Python's built-in ``hash()`` for a fast, stable mapping that does not
    require a lookup table to be populated at startup.

    Args:
        class_name: String class label (e.g. "car", "traffic light").

    Returns:
        (B, G, R) tuple of uint8 values from _COLORS.
    """
    return _COLORS[hash(class_name) % len(_COLORS)]


def draw_detections(image, detections):
    """Draw bounding boxes and confidence labels onto an image.

    For each detection, draws:
      - A 2-pixel colored rectangle around the bounding box.
      - A filled background rectangle for the text label (improves legibility).
      - The class name and confidence score as black text on the background.

    Args:
        image:      BGR numpy array (H, W, 3).  Not mutated.
        detections: Iterable of objects with attributes:
                      class_name (str), confidence (float),
                      xmin, ymin, xmax, ymax (int or float, pixel coordinates).

    Returns:
        BGR numpy array — annotated copy of the input image.
    """
    annotated = image.copy()

    for det in detections:
        # Choose a consistent color for this class.
        color = _get_color(det.class_name)
        pt1 = (int(det.xmin), int(det.ymin))
        pt2 = (int(det.xmax), int(det.ymax))

        # Bounding box outline.
        cv2.rectangle(annotated, pt1, pt2, color, 2)

        # ── Label background + text ───────────────────────────────────────────
        label = "{} {:.2f}".format(det.class_name, det.confidence)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw filled background rectangle just above the bounding box top-left.
        cv2.rectangle(annotated,
                      (pt1[0],      pt1[1] - th - baseline - 4),
                      (pt1[0] + tw, pt1[1]),
                      color, -1)  # -1 = filled

        # Black text on the colored background for maximum contrast.
        cv2.putText(annotated, label,
                    (pt1[0], pt1[1] - baseline - 2),
                    font, font_scale, (0, 0, 0), thickness)

    return annotated
