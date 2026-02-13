#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from rostopic import get_topic_type
from sensor_msgs.msg import Image, CompressedImage
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from yolov10_ros.visualization import draw_detections


class YOLOv10Detector:
    def __init__(self):
        # Load ROS params
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.5)
        self.device = str(rospy.get_param("~device", "0"))
        self.inference_size = rospy.get_param("~inference_size", 640)
        self.view_image = rospy.get_param("~view_image", False)
        self.publish_annotated = rospy.get_param("~publish_annotated", False)
        self.half = rospy.get_param("~half", False)
        weights = rospy.get_param("~weights", "")

        # Load YOLOv10 model via ultralytics API
        try:
            from ultralytics import YOLOv10
            self.model = YOLOv10(weights)
            rospy.loginfo("Loaded model with ultralytics.YOLOv10")
        except ImportError:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            rospy.loginfo("Loaded model with ultralytics.YOLO")

        rospy.loginfo("Model classes: %s", self.model.names)

        # Auto-detect input topic type (Image vs CompressedImage)
        input_image_topic = rospy.get_param("~input_image_topic", "/camera/color/image_raw")
        input_image_type, input_image_topic, _ = get_topic_type(input_image_topic, blocking=True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        msg_type = CompressedImage if self.compressed_input else Image
        self.image_sub = rospy.Subscriber(
            input_image_topic, msg_type, self.callback,
            queue_size=1, buff_size=2**24
        )
        rospy.loginfo("Subscribed to %s [%s]", input_image_topic, input_image_type)

        # Publishers
        # Lazy import of our generated messages
        from yolov10_ros.msg import Detection, DetectionArray
        self.Detection = Detection
        self.DetectionArray = DetectionArray

        output_topic = rospy.get_param("~output_topic", "/yolov10/detections")
        self.det_pub = rospy.Publisher(output_topic, DetectionArray, queue_size=10)

        output_image_topic = rospy.get_param("~output_image_topic", "/yolov10/image_raw")
        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=1)

        if self.publish_annotated:
            annotated_image_topic = rospy.get_param("~annotated_image_topic", "/yolov10/annotated")
            self.annotated_pub = rospy.Publisher(annotated_image_topic, Image, queue_size=1)

    @torch.no_grad()
    def callback(self, msg):
        # Convert ROS message to cv2 BGR image
        im0 = imgmsg_to_cv2(msg)

        # Run YOLOv10 inference (handles preprocessing internally)
        results = self.model.predict(
            source=im0,
            conf=self.conf_thres,
            imgsz=self.inference_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        # Parse results
        det_array = self.DetectionArray()
        det_array.header = msg.header

        boxes = results[0].boxes
        names = results[0].names

        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_idx = int(boxes.cls[i].cpu())

                det = self.Detection()
                det.class_name = names[cls_idx]
                det.confidence = conf
                det.xmin = int(xyxy[0])
                det.ymin = int(xyxy[1])
                det.xmax = int(xyxy[2])
                det.ymax = int(xyxy[3])
                det_array.detections.append(det)

        # Publish detections
        self.det_pub.publish(det_array)

        # Publish source image for classifier nodes to crop from
        self.image_pub.publish(cv2_to_imgmsg(im0))

        # Annotated image
        if self.publish_annotated or self.view_image:
            annotated = draw_detections(im0, det_array.detections)
            if self.publish_annotated:
                self.annotated_pub.publish(cv2_to_imgmsg(annotated))
            if self.view_image:
                cv2.imshow("YOLOv10 Detections", annotated)
                cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("yolov10_detector", anonymous=True)
    detector = YOLOv10Detector()
    rospy.spin()
