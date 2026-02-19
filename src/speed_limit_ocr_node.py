#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, crop_roi

# Valid US speed limits in mph — anything outside this set is rejected
_VALID_SPEEDS = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}


def _preprocess_crop(crop, min_height):
    """Upscale, convert to grayscale, and apply CLAHE for better OCR."""
    h, w = crop.shape[:2]
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _parse_speed(ocr_results, conf_threshold):
    """Extract a valid speed value from EasyOCR results.

    Returns (speed_int, confidence) or (None, 0) if nothing valid found.
    """
    best_speed = None
    best_conf = 0.0
    for (_, text, conf) in ocr_results:
        if conf < conf_threshold:
            continue
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            continue
        try:
            value = int(digits)
        except ValueError:
            continue
        if value in _VALID_SPEEDS and conf > best_conf:
            best_speed = value
            best_conf = conf
    return best_speed, best_conf


class SpeedLimitOCRNode:
    def __init__(self):
        # ROS params
        self.roi_padding = rospy.get_param("~roi_padding", 10)
        self.min_crop_height = rospy.get_param("~min_crop_height", 64)
        self.conf_threshold = rospy.get_param("~ocr_confidence_threshold", 0.5)
        use_gpu = str(rospy.get_param("~device", "0")).isdigit()

        sign_topic = rospy.get_param("~sign_topic", "/perception/traffic_signs")
        img_topic = rospy.get_param("~image_topic", "/yolov10/image_raw")
        output_topic = rospy.get_param("~output_topic", "/perception/speed_limit")

        # Initialize EasyOCR once — expensive to load
        import easyocr
        rospy.loginfo("Initializing EasyOCR reader (first run downloads ~100MB)...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        rospy.loginfo("EasyOCR ready")

        # Lazy import generated messages
        from yolov10_ros.msg import ClassifiedSignArray, SpeedLimit, SpeedLimitArray
        self.SpeedLimit = SpeedLimit
        self.SpeedLimitArray = SpeedLimitArray

        # Synchronized subscribers
        sign_sub = message_filters.Subscriber(sign_topic, ClassifiedSignArray)
        img_sub = message_filters.Subscriber(img_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sign_sub, img_sub], queue_size=10, slop=0.1
        )
        sync.registerCallback(self.callback)

        self.pub = rospy.Publisher(output_topic, SpeedLimitArray, queue_size=10)
        rospy.loginfo("Speed limit OCR node ready")

    def callback(self, sign_msg, img_msg):
        speed_signs = [s for s in sign_msg.signs if s.sign_type == "speed_limit"]
        if not speed_signs:
            return

        image = imgmsg_to_cv2(img_msg)
        out_msg = self.SpeedLimitArray()
        out_msg.header = sign_msg.header

        for sign in speed_signs:
            crop = crop_roi(image, sign.xmin, sign.ymin, sign.xmax, sign.ymax,
                            padding=self.roi_padding)
            if crop.size == 0:
                continue

            processed = _preprocess_crop(crop, self.min_crop_height)
            ocr_results = self.reader.readtext(
                processed, allowlist='0123456789', detail=1
            )

            speed, conf = _parse_speed(ocr_results, self.conf_threshold)
            if speed is None:
                rospy.logdebug("OCR found no valid speed limit in crop")
                continue

            sl = self.SpeedLimit()
            sl.speed = speed
            sl.display = "speed limit: {}".format(speed)
            sl.ocr_confidence = conf
            sl.xmin = sign.xmin
            sl.ymin = sign.ymin
            sl.xmax = sign.xmax
            sl.ymax = sign.ymax
            out_msg.speed_limits.append(sl)
            rospy.loginfo(sl.display)

        if out_msg.speed_limits:
            self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("speed_limit_ocr", anonymous=True)
    node = SpeedLimitOCRNode()
    rospy.spin()
