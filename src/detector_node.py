#!/usr/bin/env python3
"""
YOLOv10 Detector Node
=====================
ROS node that runs a YOLOv10 object detector on an incoming camera stream and
publishes structured bounding-box detections for downstream classifier nodes.

Subscriptions
-------------
  ~input_image_topic  (sensor_msgs/Image or sensor_msgs/CompressedImage)
      Raw camera frames.  Topic type is detected automatically at startup via
      rostopic — no manual configuration needed.

Publications
------------
  ~output_topic            (yolov10_ros/DetectionArray)   default: /yolov10/detections
      Per-frame list of bounding boxes with class name and confidence.
  ~output_image_topic      (sensor_msgs/Image)            default: /yolov10/image_raw
      The decoded source frame re-published as a plain Image so downstream
      nodes (classifiers, OCR) can crop ROIs without their own camera subscriber.
  ~annotated_image_topic   (sensor_msgs/Image)            default: /yolov10/annotated
      Optional debug view with boxes drawn.  Only published when
      ``publish_annotated`` is true.

ROS Parameters
--------------
  ~weights               (str)   Path to .pt weights file (YOLOv10 or YOLO).
  ~confidence_threshold  (float) Minimum detector confidence to include a box.
                                 Default: 0.5
  ~inference_size        (int)   Input resolution passed to the model (square).
                                 Default: 640
  ~device                (str)   CUDA device index ("0") or "cpu".
                                 Default: "0"
  ~half                  (bool)  Enable FP16 inference for speed on GPU.
                                 Default: false
  ~input_image_topic     (str)   Camera topic to subscribe to.
  ~view_image            (bool)  Pop up an OpenCV window with annotations.
                                 Default: false
  ~publish_annotated     (bool)  Publish annotated image to ROS topic.
                                 Default: false

Notes
-----
- The node gracefully falls back from ``ultralytics.YOLOv10`` to
  ``ultralytics.YOLO`` if the pinned YOLOv10 class is not available.
- The detector model was trained on BDD100K and recognises the standard
  BDD driving classes (car, truck, pedestrian, traffic light, traffic sign,
  motorcycle, bicycle, bus, rider).
- queue_size=1 + buff_size=2**24 on the image subscriber drops stale frames
  rather than building an unbounded backlog when inference is slower than
  the camera rate.
"""

import rospy
import cv2
import torch
import numpy as np
from rostopic import get_topic_type
from sensor_msgs.msg import Image, CompressedImage
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from yolov10_ros.visualization import draw_detections


class YOLOv10Detector:
    """ROS node wrapper around a YOLOv10 object detector.

    Converts incoming camera frames (Image or CompressedImage) to BGR arrays,
    runs YOLOv10 inference, and publishes a DetectionArray plus the source
    image on every frame so downstream classifier nodes can crop ROIs
    without needing their own camera subscription.
    """

    def __init__(self):
        # ── ROS parameters ────────────────────────────────────────────────────
        # Confidence threshold: detections below this score are discarded.
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.5)
        # Device: "0" → cuda:0, "1" → cuda:1, "cpu" → CPU.
        self.device = str(rospy.get_param("~device", "0"))
        # Input image size passed to the model preprocessor (square, in pixels).
        self.inference_size = rospy.get_param("~inference_size", 640)
        # Display a live OpenCV window (useful for quick field checks).
        self.view_image = rospy.get_param("~view_image", False)
        # Publish an annotated image topic for rviz / rqt_image_view.
        self.publish_annotated = rospy.get_param("~publish_annotated", False)
        # FP16 mode for roughly 2× throughput on Ampere GPUs.
        self.half = rospy.get_param("~half", False)
        weights = rospy.get_param("~weights", "")

        # ── Model loading ─────────────────────────────────────────────────────
        # Try the pinned YOLOv10 class first (ultralytics ≥ 8.1) and fall back
        # to the generic YOLO wrapper for older installs.  Both expose the same
        # .predict() API, so no other code needs to change.
        try:
            from ultralytics import YOLOv10
            self.model = YOLOv10(weights)
            rospy.loginfo("Loaded model with ultralytics.YOLOv10")
        except ImportError:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            rospy.loginfo("Loaded model with ultralytics.YOLO")

        rospy.loginfo("Model classes: %s", self.model.names)

        # ── Input subscriber ──────────────────────────────────────────────────
        # Query the ROS master for the topic type so we accept both raw Image
        # and CompressedImage without manual reconfiguration.
        input_image_topic = rospy.get_param("~input_image_topic", "/camera/color/image_raw")
        input_image_type, input_image_topic, _ = get_topic_type(input_image_topic, blocking=True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        msg_type = CompressedImage if self.compressed_input else Image
        # queue_size=1: only process the most recent frame (drop stale ones).
        # buff_size=2**24 (16 MB): avoid buffer overflow on high-res streams.
        self.image_sub = rospy.Subscriber(
            input_image_topic, msg_type, self.callback,
            queue_size=1, buff_size=2**24
        )
        rospy.loginfo("Subscribed to %s [%s]", input_image_topic, input_image_type)

        # ── Publishers ────────────────────────────────────────────────────────
        # Lazy-import generated message types so the node can start before
        # catkin has compiled them (avoids import-time failures in devel builds).
        from yolov10_ros.msg import Detection, DetectionArray
        self.Detection = Detection
        self.DetectionArray = DetectionArray

        output_topic = rospy.get_param("~output_topic", "/yolov10/detections")
        self.det_pub = rospy.Publisher(output_topic, DetectionArray, queue_size=10)

        # Re-publish the decoded source frame so classifiers can crop without
        # needing their own camera subscription.
        output_image_topic = rospy.get_param("~output_image_topic", "/yolov10/image_raw")
        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=1)

        if self.publish_annotated:
            annotated_image_topic = rospy.get_param("~annotated_image_topic", "/yolov10/annotated")
            self.annotated_pub = rospy.Publisher(annotated_image_topic, Image, queue_size=1)

    @torch.no_grad()
    def callback(self, msg):
        """Process one camera frame: run detection and publish results.

        Args:
            msg: sensor_msgs/Image or sensor_msgs/CompressedImage from the camera.

        Publishes:
            DetectionArray on the configured output topic.
            Image (source frame) on the output image topic.
            Image (annotated) on the annotated topic if publish_annotated is true.
        """
        # ── Decode ────────────────────────────────────────────────────────────
        # imgmsg_to_cv2 handles both Image and CompressedImage transparently.
        im0 = imgmsg_to_cv2(msg)

        # ── Inference ─────────────────────────────────────────────────────────
        # ultralytics handles all preprocessing (resize, normalize, letterbox)
        # internally.  Results are returned in original image pixel coordinates
        # — no manual scaling / inverse-letterbox needed.
        results = self.model.predict(
            source=im0,
            conf=self.conf_thres,
            imgsz=self.inference_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        # ── Parse results → DetectionArray ────────────────────────────────────
        det_array = self.DetectionArray()
        # Preserve the original frame timestamp for time-synchronisation with
        # downstream classifier nodes.
        det_array.header = msg.header

        boxes = results[0].boxes   # Boxes object (may be None if no detections)
        names = results[0].names   # {class_idx: class_name} dict

        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                # xyxy coords are in the original image frame (pixels, float).
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

        # ── Publish ───────────────────────────────────────────────────────────
        self.det_pub.publish(det_array)

        # Republish the source image so light/sign/OCR nodes can crop from it
        # without needing their own camera subscriber.
        img_msg = cv2_to_imgmsg(im0)
        img_msg.header = msg.header  # preserve original camera timestamp for ApproximateTimeSynchronizer
        self.image_pub.publish(img_msg)

        # Optional annotated image (for rviz / rqt_image_view debugging).
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
