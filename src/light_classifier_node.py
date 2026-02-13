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

# Class mapping from training (ImageFolder alphabetical order)
_DEFAULT_CLASSES = {
    0: "green_left",
    1: "green_light",
    2: "red_left",
    3: "red_light",
    4: "yellow_left",
    5: "yellow_light",
}


class LightClassifierNode:
    def __init__(self):
        # Load ROS params
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.5)
        self.device = str(rospy.get_param("~device", "0"))
        self.target_class = rospy.get_param("~target_class", "traffic light")
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
        rospy.loginfo("Light classifier loaded on %s (%d classes)",
                      self.torch_device, num_classes)

        # Lazy import generated messages
        from yolov10_ros.msg import DetectionArray, ClassifiedLight, ClassifiedLightArray
        self.ClassifiedLight = ClassifiedLight
        self.ClassifiedLightArray = ClassifiedLightArray

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
        output_topic = rospy.get_param("~output_topic", "/perception/traffic_lights")
        self.pub = rospy.Publisher(output_topic, ClassifiedLightArray, queue_size=10)

        rospy.loginfo("Light classifier node ready — filtering for '%s'", self.target_class)

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

        out_msg = self.ClassifiedLightArray()
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

            cl = self.ClassifiedLight()
            cl.state = self.class_map.get(idx, "unknown")
            cl.state_confidence = conf
            cl.xmin = det.xmin
            cl.ymin = det.ymin
            cl.xmax = det.xmax
            cl.ymax = det.ymax
            out_msg.lights.append(cl)

        if out_msg.lights:
            self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("light_classifier", anonymous=True)
    node = LightClassifierNode()
    rospy.spin()
