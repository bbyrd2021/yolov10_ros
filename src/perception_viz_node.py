#!/usr/bin/env python3
"""
Perception Visualization Node
==============================
Combines all pipeline outputs onto a single annotated image and publishes it
to /perception/annotated for viewing in rqt_image_view or rviz.

What gets drawn (in order, so classifiers overwrite plain detector boxes):
  1. YOLOv10 detector boxes   — class name + confidence, per-class color
  2. Classified traffic lights — state label + confidence, colored by state
  3. Classified traffic signs  — sign_type + confidence, sky-blue
  4. Speed limit OCR values    — "SPEED N mph", red

Architecture
------------
Uses a cache approach: each subscriber stores the latest message, and the
image callback redraws everything from cache on every new frame.  This means
classifier/OCR results from the previous frame are shown until updated —
which is correct behaviour since those nodes may run slower than the camera.

Subscriptions
-------------
  ~detection_topic  (yolov10_ros/DetectionArray)       default: /yolov10/detections
  ~image_topic      (sensor_msgs/Image)                 default: /yolov10/image_raw
  ~light_topic      (yolov10_ros/ClassifiedLightArray)  default: /perception/traffic_lights
  ~sign_topic       (yolov10_ros/ClassifiedSignArray)   default: /perception/traffic_signs
  ~speed_topic      (yolov10_ros/SpeedLimitArray)       default: /perception/speed_limit

Publications
------------
  ~output_topic     (sensor_msgs/Image)                 default: /perception/annotated
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from yolov10_ros.visualization import CLASS_COLORS

# ── Additional color maps (BGR) ───────────────────────────────────────────────

# Traffic light state → box color
_LIGHT_COLORS = {
    'green_light':  ( 30, 200,  30),
    'green_left':   ( 30, 200,  30),
    'red_light':    (  0,   0, 220),
    'red_left':     (  0,   0, 220),
    'yellow_light': (  0, 200, 220),
    'yellow_left':  (  0, 200, 220),
}

_SIGN_COLOR    = (255, 180,   0)   # sky-blue for all sign types
_SPEED_COLOR   = (  0,   0, 255)   # red — highest visual priority
_FALLBACK      = (200, 200, 200)   # grey for unlisted classes

# Classes drawn by downstream classifiers — skipped in the raw YOLO pass to
# avoid a double-labelled box (classifier box is drawn on top in steps 2-3).
_CLASSIFIER_CLASSES = frozenset({'traffic light', 'traffic sign'})


# ── Drawing helper ────────────────────────────────────────────────────────────

def _draw_box(img, x1, y1, x2, y2, label, color, thickness=2):
    """Draw a colored bounding box with a filled label background.

    Args:
        img:       BGR numpy array, mutated in place.
        x1,y1:     Top-left corner (pixels).
        x2,y2:     Bottom-right corner (pixels).
        label:     Text string drawn above the box.
        color:     BGR tuple for the box outline and label background.
        thickness: Outline thickness in pixels.
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)

    # Filled background rectangle just above the box top edge.
    cv2.rectangle(img, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - bl - 2), font, scale, (0, 0, 0), thick)


# ── Node class ────────────────────────────────────────────────────────────────

class PerceptionVizNode:
    """Combines all pipeline outputs onto one annotated image."""

    def __init__(self):
        # Topic names — overridable via ROS params.
        det_topic   = rospy.get_param('~detection_topic', '/yolov10/detections')
        img_topic   = rospy.get_param('~image_topic',     '/yolov10/image_raw')
        light_topic = rospy.get_param('~light_topic',     '/perception/traffic_lights')
        sign_topic  = rospy.get_param('~sign_topic',      '/perception/traffic_signs')
        speed_topic = rospy.get_param('~speed_topic',     '/perception/speed_limit')
        out_topic   = rospy.get_param('~output_topic',    '/perception/annotated')

        da_topic    = rospy.get_param('~drivable_area_topic', '/perception/drivable_area')
        ll_topic    = rospy.get_param('~lane_lines_topic',   '/perception/lane_lines')

        # Message caches — updated by their respective subscribers.
        self._detections = None
        self._lights     = None
        self._signs      = None
        self._speed      = None
        self._da_mask    = None
        self._ll_mask    = None

        # Import message types lazily so this node starts after catkin build
        # regenerates the Python message modules.
        from yolov10_ros.msg import (
            DetectionArray, ClassifiedLightArray,
            ClassifiedSignArray, SpeedLimitArray,
        )

        rospy.Subscriber(det_topic,   DetectionArray,       self._cb_det,    queue_size=1)
        rospy.Subscriber(light_topic, ClassifiedLightArray, self._cb_lights, queue_size=1)
        rospy.Subscriber(sign_topic,  ClassifiedSignArray,  self._cb_signs,  queue_size=1)
        rospy.Subscriber(speed_topic, SpeedLimitArray,      self._cb_speed,  queue_size=1)
        rospy.Subscriber(da_topic,    Image,                self._cb_da,     queue_size=1)
        rospy.Subscriber(ll_topic,    Image,                self._cb_ll,     queue_size=1)

        # Image subscriber triggers the draw pass on every frame.
        rospy.Subscriber(img_topic, Image, self._cb_image,
                         queue_size=1, buff_size=2**24)

        self._pub = rospy.Publisher(out_topic, Image, queue_size=1)
        rospy.loginfo("Perception visualizer ready → %s", out_topic)

    # ── Cache callbacks ───────────────────────────────────────────────────────

    def _cb_det(self, msg):
        self._detections = msg

    def _cb_lights(self, msg):
        self._lights = msg

    def _cb_signs(self, msg):
        self._signs = msg

    def _cb_speed(self, msg):
        self._speed = msg

    def _cb_da(self, msg):
        self._da_mask = imgmsg_to_cv2(msg, encoding='mono8')

    def _cb_ll(self, msg):
        self._ll_mask = imgmsg_to_cv2(msg, encoding='mono8')

    # ── Draw callback ─────────────────────────────────────────────────────────

    def _cb_image(self, msg):
        """Draw all cached results onto the current frame and publish."""
        frame = imgmsg_to_cv2(msg)
        out   = frame.copy()

        # Invalidate downstream classifier caches when the detector no longer
        # sees the parent class.  The detector publishes on every frame (even
        # when empty), so self._detections is always current.
        if self._detections is not None:
            det_classes = {d.class_name for d in self._detections.detections}
            if 'traffic light' not in det_classes:
                self._lights = None
            if 'traffic sign' not in det_classes:
                self._signs = None
                self._speed = None

        # 0. Segmentation overlays — blended behind all bounding boxes.
        #    DA: semi-transparent blue.  LL: semi-transparent green.
        if self._da_mask is not None and self._da_mask.shape == frame.shape[:2]:
            overlay = out.copy()
            overlay[self._da_mask > 127] = (255, 0, 0)
            out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)

        if self._ll_mask is not None and self._ll_mask.shape == frame.shape[:2]:
            overlay = out.copy()
            overlay[self._ll_mask > 127] = (0, 255, 0)
            out = cv2.addWeighted(out, 0.70, overlay, 0.30, 0)

        # 1. YOLO detector boxes — skip classes owned by downstream classifiers
        #    to prevent a double-labelled box (classifier draws its own in
        #    steps 2-3 with richer information).
        if self._detections:
            for det in self._detections.detections:
                if det.class_name in _CLASSIFIER_CLASSES:
                    continue
                color = CLASS_COLORS.get(det.class_name, _FALLBACK)
                label = '{} {:.2f}'.format(det.class_name, det.confidence)
                _draw_box(out, det.xmin, det.ymin, det.xmax, det.ymax,
                          label, color, thickness=2)

        # 2. Classified traffic lights.
        if self._lights:
            for lt in self._lights.lights:
                color = _LIGHT_COLORS.get(lt.state, _FALLBACK)
                label = '{} {:.2f}'.format(lt.state, lt.state_confidence)
                _draw_box(out, lt.xmin, lt.ymin, lt.xmax, lt.ymax,
                          label, color, thickness=3)

        # 3. Classified traffic signs.
        #    Speed-limit signs are rendered with the OCR value when available
        #    ("SPEED 35 mph") or a placeholder ("SPEED ? mph") when OCR has
        #    not yet returned a result.  All other sign types show the
        #    classifier label and confidence.
        if self._signs:
            speed_lookup = {}
            if self._speed:
                for sl in self._speed.speed_limits:
                    speed_lookup[(sl.xmin, sl.ymin, sl.xmax, sl.ymax)] = sl.speed

            for sg in self._signs.signs:
                if sg.sign_type == 'speed_limit':
                    n = speed_lookup.get((sg.xmin, sg.ymin, sg.xmax, sg.ymax))
                    label = 'SPEED {} mph'.format(n) if n is not None else 'SPEED ? mph'
                    _draw_box(out, sg.xmin, sg.ymin, sg.xmax, sg.ymax,
                              label, _SPEED_COLOR, thickness=4)
                else:
                    label = '{} {:.2f}'.format(sg.sign_type, sg.type_confidence)
                    _draw_box(out, sg.xmin, sg.ymin, sg.xmax, sg.ymax,
                              label, _SIGN_COLOR, thickness=3)

        self._pub.publish(cv2_to_imgmsg(out, 'bgr8'))


if __name__ == '__main__':
    rospy.init_node('perception_viz', anonymous=False)
    PerceptionVizNode()
    rospy.spin()
