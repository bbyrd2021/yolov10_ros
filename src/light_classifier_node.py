#!/usr/bin/env python3
"""
Traffic Light Classifier Node
==============================
Downstream ROS node that classifies the *state* of traffic lights detected by
the YOLOv10 detector node.

Architecture
------------
The detector node publishes raw bounding boxes for every object class including
``traffic light``.  This node subscribes to those detections **and** the
corresponding decoded image, filters for traffic-light boxes, crops the ROI,
and runs a fine-tuned EfficientNet B0 classifier to determine the light state
(green / red / yellow, and left-arrow variants).

Subscriptions
-------------
  ~detection_topic  (yolov10_ros/DetectionArray)  default: /yolov10/detections
      Bounding boxes from the detector node.
  ~image_topic      (sensor_msgs/Image)            default: /yolov10/image_raw
      Decoded source frame republished by the detector node.

  Both topics are time-synchronised via message_filters.ApproximateTimeSynchronizer
  (slop = 0.1 s) so crops are always taken from the correct frame.

Publications
------------
  ~output_topic  (yolov10_ros/ClassifiedLightArray)  default: /perception/traffic_lights
      One ClassifiedLight message per confirmed traffic light, carrying the
      predicted state string, its softmax confidence, and the bounding box.

ROS Parameters
--------------
  ~weights               (str)   Path to EfficientNet B0 .pth checkpoint.
  ~confidence_threshold  (float) Minimum classifier softmax confidence to publish.
                                 Default: 0.5
  ~device                (str)   CUDA device index ("0") or "cpu".  Default: "0"
  ~target_class          (str)   Detector class name to filter for.
                                 Default: "traffic light"
  ~roi_padding           (int)   Extra pixels added around each bounding box
                                 before cropping (helps capture full housing).
                                 Default: 5
  ~class_names           (list)  Override the default class mapping via YAML
                                 (list in alphabetical training order).

Class Mapping (default — ImageFolder alphabetical)
---------------------------------------------------
  0  green_left    — green arrow pointing left
  1  green_light   — solid green
  2  red_left      — red arrow pointing left
  3  red_light     — solid red
  4  yellow_left   — yellow arrow pointing left
  5  yellow_light  — solid yellow

Notes
-----
- The EfficientNet B0 checkpoint stores weights under the key
  ``model_state_dict``.  The classifier head is replaced to match the number
  of output classes before loading.
- The EfficientNet input is normalised with ImageNet mean/std
  ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) after scaling to [0, 1].
"""

import rospy
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, crop_roi


# ── EfficientNet B0 preprocessing constants ────────────────────────────────────
# All EfficientNet variants expect 224 × 224 RGB input normalised to ImageNet
# statistics.  These constants must match those used during training exactly.
_INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet RGB mean
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # ImageNet RGB std

# ── Default class mapping ─────────────────────────────────────────────────────
# Derived from the training dataset directory structure (ImageFolder sorts class
# folders alphabetically, so the index order is fixed by folder name).
_DEFAULT_CLASSES = {
    0: "green_left",    # Green arrow (left turn permitted)
    1: "green_light",   # Solid green (go)
    2: "red_left",      # Red arrow (left turn prohibited)
    3: "red_light",     # Solid red (stop)
    4: "yellow_left",   # Yellow arrow (caution / left turn clearing)
    5: "yellow_light",  # Solid yellow (caution / prepare to stop)
}


class LightClassifierNode:
    """EfficientNet B0 traffic-light state classifier wrapped as a ROS node.

    Processes each DetectionArray message, crops every ``traffic light`` ROI
    from the synchronised image, runs the classifier, and publishes a
    ClassifiedLightArray containing only high-confidence predictions.
    """

    def __init__(self):
        # ── ROS parameters ────────────────────────────────────────────────────
        # Drop predictions whose softmax probability is below this threshold.
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.5)
        # Device string: digit → CUDA device, otherwise passed to torch.device.
        self.device = str(rospy.get_param("~device", "0"))
        # Only boxes whose class_name matches this string are passed to the
        # classifier.  Defaults to the BDD100K class name for traffic lights.
        self.target_class = rospy.get_param("~target_class", "traffic light")
        # Pixels added on all sides of the bounding box before cropping.
        # A small padding helps capture the full light housing.
        self.roi_padding = rospy.get_param("~roi_padding", 5)
        weights = rospy.get_param("~weights", "")

        # ── Class mapping ─────────────────────────────────────────────────────
        # Allow overriding the default mapping via a YAML list param so the
        # node can be retrained for different label sets without code changes.
        class_list = rospy.get_param("~class_names", None)
        if class_list:
            self.class_map = {i: name for i, name in enumerate(class_list)}
        else:
            self.class_map = _DEFAULT_CLASSES

        # ── Model loading ─────────────────────────────────────────────────────
        # Build the torch device from the string param:
        #   "0"   → cuda:0
        #   "cpu" → cpu
        self.torch_device = torch.device(
            "cuda:{}".format(self.device) if self.device.isdigit() else self.device
        )
        from torchvision.models import efficientnet_b0
        # Instantiate with random weights first, then overwrite the classifier
        # head to match the number of training classes, then load the checkpoint.
        # (torch.load() alone would return a raw dict, not a callable model.)
        self.model = efficientnet_b0(weights=None)
        num_classes = len(self.class_map)
        # Replace the final linear layer: EfficientNet B0 has 1280 features
        # before the classifier head.
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )
        # Checkpoints are saved as {"epoch": ..., "model_state_dict": ..., ...}
        ckpt = torch.load(weights, map_location=self.torch_device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.torch_device)
        self.model.eval()
        rospy.loginfo("Light classifier loaded on %s (%d classes)",
                      self.torch_device, num_classes)

        # ── Lazy message imports ──────────────────────────────────────────────
        # Importing generated messages after catkin build avoids hard failures
        # when the workspace is only partially compiled.
        from yolov10_ros.msg import DetectionArray, ClassifiedLight, ClassifiedLightArray
        self.ClassifiedLight = ClassifiedLight
        self.ClassifiedLightArray = ClassifiedLightArray

        # ── Synchronized subscribers ──────────────────────────────────────────
        # ApproximateTimeSynchronizer pairs messages whose header timestamps are
        # within `slop` seconds of each other.  This guarantees that the bounding
        # boxes and the image being cropped come from the same camera frame.
        det_topic = rospy.get_param("~detection_topic", "/yolov10/detections")
        img_topic = rospy.get_param("~image_topic", "/yolov10/image_raw")

        det_sub = message_filters.Subscriber(det_topic, DetectionArray)
        img_sub = message_filters.Subscriber(img_topic, Image)

        sync = message_filters.ApproximateTimeSynchronizer(
            [det_sub, img_sub], queue_size=10, slop=0.1
        )
        sync.registerCallback(self.callback)

        # ── Publisher ─────────────────────────────────────────────────────────
        output_topic = rospy.get_param("~output_topic", "/perception/traffic_lights")
        self.pub = rospy.Publisher(output_topic, ClassifiedLightArray, queue_size=10)

        rospy.loginfo("Light classifier node ready — filtering for '%s'", self.target_class)

    def _preprocess(self, crop):
        """Resize a BGR crop to the EfficientNet input format.

        Converts the crop to RGB, scales to [0, 1], applies ImageNet
        normalisation, and returns a (1, 3, 224, 224) float32 tensor
        already on the correct device.

        Args:
            crop: H×W×3 BGR numpy array (arbitrary size).

        Returns:
            torch.Tensor of shape (1, 3, 224, 224) on self.torch_device.
        """
        img = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        # HWC → CHW → add batch dimension
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.torch_device)

    @torch.no_grad()
    def callback(self, det_msg, img_msg):
        """Handle a synchronised (DetectionArray, Image) pair.

        Filters detections for target_class, crops each ROI, classifies it,
        and publishes a ClassifiedLightArray if any high-confidence results
        are found.

        Args:
            det_msg: yolov10_ros/DetectionArray — bounding boxes from detector.
            img_msg: sensor_msgs/Image — source frame matching det_msg timestamp.
        """
        # Only process frames that contain at least one traffic-light detection.
        relevant = [d for d in det_msg.detections if d.class_name == self.target_class]
        if not relevant:
            return

        # Decode the source image once (shared across all crops in this frame).
        image = imgmsg_to_cv2(img_msg)

        out_msg = self.ClassifiedLightArray()
        # Preserve the original frame timestamp for downstream subscribers.
        out_msg.header = det_msg.header

        for det in relevant:
            # crop_roi clamps coordinates to image bounds and adds padding.
            crop = crop_roi(image, det.xmin, det.ymin, det.xmax, det.ymax,
                            padding=self.roi_padding)
            if crop.size == 0:
                # Guard against degenerate boxes (e.g., at image edges).
                continue

            # Run the classifier: softmax gives a proper probability distribution
            # so we can threshold on the top-class confidence directly.
            tensor = self._preprocess(crop)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            conf = float(conf)
            idx = int(idx)

            # Discard low-confidence predictions.
            if conf < self.conf_thres:
                continue

            cl = self.ClassifiedLight()
            cl.state = self.class_map.get(idx, "unknown")
            cl.state_confidence = conf
            # Pass bounding box through so consumers know where the light is.
            cl.xmin = det.xmin
            cl.ymin = det.ymin
            cl.xmax = det.xmax
            cl.ymax = det.ymax
            out_msg.lights.append(cl)

        # Only publish if at least one light passed the confidence gate.
        if out_msg.lights:
            self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("light_classifier", anonymous=True)
    node = LightClassifierNode()
    rospy.spin()
