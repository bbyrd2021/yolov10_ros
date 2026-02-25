#!/usr/bin/env python3
"""
Lane Segmentation Node
======================
Runs TwinLiteNet on every decoded camera frame and publishes two binary
segmentation masks:

  /perception/drivable_area  — mono8, 255 = drivable surface
  /perception/lane_lines     — mono8, 255 = lane marking

Subscribes to /yolov10/image_raw (already published by the detector node)
so no additional camera subscription is needed.

Subscriptions
-------------
  ~image_topic  (sensor_msgs/Image)  default: /yolov10/image_raw

Publications
------------
  ~drivable_area_topic  (sensor_msgs/Image, mono8)
      Binary mask: 255 = drivable area, 0 = background.
  ~lane_lines_topic     (sensor_msgs/Image, mono8)
      Binary mask: 255 = lane line, 0 = background.

ROS Parameters
--------------
  ~weights             (str)   Path to twinlitenet_best.pth checkpoint.
  ~device              (str)   CUDA index ("0") or "cpu".  Default: "0"
  ~image_topic         (str)   Override input topic.
  ~drivable_area_topic (str)   Override drivable area output topic.
  ~lane_lines_topic    (str)   Override lane lines output topic.

Preprocessing (matches training convention)
-------------------------------------------
  1. Resize to 640 × 360 (W × H)
  2. BGR → RGB
  3. HWC → CHW, contiguous
  4. float32 / 255.0  →  [0, 1]
  5. Unsqueeze batch dim → (1, 3, 360, 640)

Postprocessing
--------------
  • argmax over 2-class logits → binary label map (0 or 1)
  • multiply by 255 → uint8 mask at 640 × 360
  • resize back to original frame resolution with nearest-neighbour interpolation
"""

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from yolov10_ros.twinlitenet import TwinLiteNet

# Fixed input resolution the model was trained on.
_IN_W, _IN_H = 640, 360


def _preprocess(bgr_frame):
    """Resize, convert to RGB tensor, normalise to [0, 1].

    Args:
        bgr_frame: uint8 BGR numpy array (H, W, 3).

    Returns:
        Float32 CUDA tensor of shape (1, 3, 360, 640).
    """
    img = cv2.resize(bgr_frame, (_IN_W, _IN_H))
    img = img[:, :, ::-1]               # BGR → RGB
    img = img.transpose(2, 0, 1)        # HWC → CHW
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0)          # (1, 3, 360, 640)


def _postprocess(logits, orig_h, orig_w):
    """Convert 2-class logits to a resized uint8 binary mask.

    Args:
        logits:  Tensor (1, 2, 360, 640) — raw model output.
        orig_h:  Target height in pixels.
        orig_w:  Target width in pixels.

    Returns:
        uint8 numpy array (orig_h, orig_w): 255 = foreground, 0 = background.
    """
    _, pred = torch.max(logits, dim=1)          # (1, 360, 640)
    mask = pred.byte().cpu().numpy()[0] * 255   # uint8, model resolution
    if mask.shape != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


class LaneSegNode:
    """ROS node wrapper for TwinLiteNet drivable area and lane segmentation."""

    def __init__(self):
        weights   = rospy.get_param('~weights', '')
        device_id = str(rospy.get_param('~device', '0'))

        img_topic = rospy.get_param('~image_topic',
                                    '/yolov10/image_raw')
        da_topic  = rospy.get_param('~drivable_area_topic',
                                    '/perception/drivable_area')
        ll_topic  = rospy.get_param('~lane_lines_topic',
                                    '/perception/lane_lines')

        # ── Device selection ──────────────────────────────────────────────────
        if device_id.isdigit() and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(device_id))
        else:
            self.device = torch.device('cpu')
        rospy.loginfo("Lane seg using device: %s", self.device)

        # ── Model loading ─────────────────────────────────────────────────────
        self.model = TwinLiteNet()

        if not weights:
            rospy.logfatal("~weights parameter is required")
            raise RuntimeError("~weights not set")

        state = torch.load(weights, map_location='cpu')
        # Weights saved with DataParallel — strip the 'module.' prefix.
        clean = {k.replace('module.', '', 1): v for k, v in state.items()}
        self.model.load_state_dict(clean)
        self.model.to(self.device)
        self.model.eval()
        rospy.loginfo("TwinLiteNet loaded from %s", weights)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_da = rospy.Publisher(da_topic, Image, queue_size=1)
        self.pub_ll = rospy.Publisher(ll_topic, Image, queue_size=1)

        # ── Subscriber ───────────────────────────────────────────────────────
        rospy.Subscriber(img_topic, Image, self.callback,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo("Lane seg node ready (DA → %s  LL → %s)", da_topic, ll_topic)

    @torch.no_grad()
    def callback(self, msg):
        frame = imgmsg_to_cv2(msg)
        orig_h, orig_w = frame.shape[:2]

        tensor = _preprocess(frame).to(self.device)
        da_logits, ll_logits = self.model(tensor)

        da_mask = _postprocess(da_logits, orig_h, orig_w)
        ll_mask = _postprocess(ll_logits, orig_h, orig_w)

        stamp = msg.header.stamp

        da_msg = cv2_to_imgmsg(da_mask, 'mono8')
        da_msg.header.stamp    = stamp
        da_msg.header.frame_id = msg.header.frame_id

        ll_msg = cv2_to_imgmsg(ll_mask, 'mono8')
        ll_msg.header.stamp    = stamp
        ll_msg.header.frame_id = msg.header.frame_id

        self.pub_da.publish(da_msg)
        self.pub_ll.publish(ll_msg)


if __name__ == '__main__':
    rospy.init_node('lane_seg', anonymous=False)
    LaneSegNode()
    rospy.spin()
