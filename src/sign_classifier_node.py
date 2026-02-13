#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, crop_roi


# EfficientNet B0 input size
_INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class mapping from training
_DEFAULT_CLASSES = {
    0: "detour",
    1: "do_not_enter",
    2: "go_straight_only",
    3: "no_left_turn",
    4: "no_right_turn",
    5: "no_straight",
    6: "no_u_turn",
    7: "pedestrian_crossing",
    8: "railroad_crossing",
    9: "roadwork",
    10: "speed_limit",
    11: "stop",
    12: "turn_left_only",
    13: "turn_right_only",
    14: "yield",
}


class SignClassifierNode:
    def __init__(self):
        # Load ROS params
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.5)
        self.device = str(rospy.get_param("~device", "0"))
        self.target_class = rospy.get_param("~target_class", "traffic sign")
        self.roi_padding = rospy.get_param("~roi_padding", 5)
        weights = rospy.get_param("~weights", "")

        # Class mapping (can be overridden via param)
        class_list = rospy.get_param("~class_names", None)
        if class_list:
            self.class_map = {i: name for i, name in enumerate(class_list)}
        else:
            self.class_map = _DEFAULT_CLASSES

        # Load EfficientNet B0 classifier from state_dict checkpoint
        self.torch_device = torch.device(
            "cuda:{}".format(self.device) if self.device.isdigit() else self.device
        )
        from torchvision.models import efficientnet_b0
        self.model = efficientnet_b0(weights=None)
        num_classes = len(self.class_map)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )
        ckpt = torch.load(weights, map_location=self.torch_device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.torch_device)
        self.model.eval()
        rospy.loginfo("Sign classifier loaded on %s (%d classes)",
                      self.torch_device, num_classes)

        # Lazy import generated messages
        from yolov10_ros.msg import DetectionArray, ClassifiedSign, ClassifiedSignArray
        self.ClassifiedSign = ClassifiedSign
        self.ClassifiedSignArray = ClassifiedSignArray

        # Synchronized subscribers
        det_topic = rospy.get_param("~detection_topic", "/yolov10/detections")
        img_topic = rospy.get_param("~image_topic", "/yolov10/image_raw")

        det_sub = message_filters.Subscriber(det_topic, DetectionArray)
        img_sub = message_filters.Subscriber(img_topic, Image)

        sync = message_filters.ApproximateTimeSynchronizer(
            [det_sub, img_sub], queue_size=10, slop=0.1
        )
        sync.registerCallback(self.callback)

        # Publisher
        output_topic = rospy.get_param("~output_topic", "/perception/traffic_signs")
        self.pub = rospy.Publisher(output_topic, ClassifiedSignArray, queue_size=10)

        rospy.loginfo("Sign classifier node ready — filtering for '%s'", self.target_class)

    def _preprocess(self, crop):
        """Resize, normalize, and convert a BGR crop to a model-ready tensor."""
        img = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.torch_device)

    @torch.no_grad()
    def callback(self, det_msg, img_msg):
        # Filter detections for target class
        relevant = [d for d in det_msg.detections if d.class_name == self.target_class]
        if not relevant:
            return

        image = imgmsg_to_cv2(img_msg)

        out_msg = self.ClassifiedSignArray()
        out_msg.header = det_msg.header

        for det in relevant:
            crop = crop_roi(image, det.xmin, det.ymin, det.xmax, det.ymax,
                            padding=self.roi_padding)
            if crop.size == 0:
                continue

            tensor = self._preprocess(crop)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            conf = float(conf)
            idx = int(idx)

            if conf < self.conf_thres:
                continue

            cs = self.ClassifiedSign()
            cs.sign_type = self.class_map.get(idx, "unknown")
            cs.type_confidence = conf
            cs.xmin = det.xmin
            cs.ymin = det.ymin
            cs.xmax = det.xmax
            cs.ymax = det.ymax
            out_msg.signs.append(cs)

        if out_msg.signs:
            self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("sign_classifier", anonymous=True)
    node = SignClassifierNode()
    rospy.spin()
