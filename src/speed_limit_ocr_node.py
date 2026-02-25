#!/usr/bin/env python3
"""
Speed Limit OCR Node
====================
Terminal ROS node in the perception pipeline.  Reads the numeric value off
speed-limit sign crops using EasyOCR and publishes structured SpeedLimit
messages for the autonomy stack.

Architecture
------------
The sign classifier node publishes ClassifiedSignArray messages on
``/perception/traffic_signs``.  This node subscribes to those classifications
*and* to the matching decoded image, isolates signs whose ``sign_type`` is
``"speed_limit"``, crops the ROI, preprocesses for OCR, and publishes a
SpeedLimitArray containing the parsed integer value and OCR confidence.

Why a separate node?
~~~~~~~~~~~~~~~~~~~~
EasyOCR requires ~100 MB of model data and takes a few seconds to initialise.
Isolating it in its own node means the detector and classifiers are unaffected
if OCR is disabled or crashes, and lets the OCR rate decouple from the camera
frame rate if needed.

Subscriptions
-------------
  ~sign_topic   (yolov10_ros/ClassifiedSignArray)  default: /perception/traffic_signs
      Sign classifications from the sign classifier node.
  ~image_topic  (sensor_msgs/Image)                default: /yolov10/image_raw
      Decoded source frame; subscribed via ApproximateTimeSynchronizer.

Publications
------------
  ~output_topic  (yolov10_ros/SpeedLimitArray)  default: /perception/speed_limit
      One SpeedLimit per successfully parsed sign, carrying the integer mph
      value, a human-readable display string, OCR confidence, and bounding box.

ROS Parameters
--------------
  ~roi_padding             (int)   Extra pixels added around the sign box before
                                   cropping.  Default: 10
  ~min_crop_height         (int)   Minimum crop height in pixels; smaller crops
                                   are upscaled with bicubic interpolation before
                                   OCR to improve character recognition.
                                   Default: 64
  ~ocr_confidence_threshold (float) Minimum EasyOCR word confidence to accept.
                                    Default: 0.5
  ~device                  (str)   CUDA device index ("0") or "cpu".
                                   Passed to easyocr.Reader(gpu=...).
  ~sign_topic              (str)   Override subscription topic.
  ~image_topic             (str)   Override image topic.
  ~output_topic            (str)   Override publication topic.

OCR Preprocessing Pipeline
---------------------------
1. Upscale crop to at least ``min_crop_height`` pixels tall (bicubic) —
   EasyOCR accuracy degrades significantly below ~50 px character height.
2. Convert to grayscale.
3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) with
   clipLimit=2.0 and tileSize=(4,4) to improve contrast on faded or
   shadowed signs.
4. Run ``reader.readtext(crop, allowlist='0123456789', detail=1)`` — the
   digit-only allowlist prevents misreads of the circular sign border or
   interior text as letters.

Speed Validation
----------------
Parsed integers are validated against the set of legal US speed limits
(mph): {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}.
Values outside this set (e.g. year numbers from a date sticker, or
partial reads like "5" from "55") are discarded.  The best-confidence
valid reading is chosen if the OCR returns multiple results.

Notes
-----
- The EasyOCR Reader object is created once in ``__init__`` — it loads CRAFT
  (text detection) and CRNN (text recognition) models on startup.  Creating it
  per-callback would add ~2 s latency on every callback invocation.
- On first run, EasyOCR downloads approximately 100 MB of model weights to
  ``~/.EasyOCR/``.
"""

import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from yolov10_ros.image_utils import imgmsg_to_cv2, crop_roi


# ── Speed limit whitelist ─────────────────────────────────────────────────────
# Standard posted US speed limits in mph.  Any OCR result outside this set is
# rejected to prevent garbage reads (e.g. partial numbers, sign serial numbers).
_VALID_SPEEDS = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}


def _preprocess_crop(crop, min_height):
    """Prepare a BGR sign crop for EasyOCR.

    Steps:
        1. Upscale if the crop is shorter than ``min_height`` (bicubic
           interpolation preserves sharpness better than linear for
           small-to-large upscaling).
        2. Convert to grayscale — EasyOCR's text detector works on single-channel.
        3. Apply CLAHE to enhance local contrast on sun-faded or shadowed signs.

    Args:
        crop:       H×W×3 BGR numpy array (the speed limit sign ROI).
        min_height: Minimum height in pixels.  Crops shorter than this are scaled
                    up proportionally.

    Returns:
        Grayscale numpy array (H'×W', uint8) ready to pass to easyocr.Reader.
    """
    h, w = crop.shape[:2]
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # CLAHE parameters: clipLimit=2.0 caps contrast amplification to prevent
    # noise from becoming dominant; tileGridSize=(4,4) handles local lighting
    # variations common on outdoor signage.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _parse_speed(ocr_results, conf_threshold):
    """Extract the best valid speed reading from EasyOCR output.

    Iterates over all OCR results, strips non-digit characters, casts to int,
    checks membership in ``_VALID_SPEEDS``, and returns the highest-confidence
    valid reading.

    Args:
        ocr_results:    List of (bbox, text, confidence) tuples from
                        easyocr.Reader.readtext(..., detail=1).
        conf_threshold: Float in [0, 1].  Readings below this confidence are
                        ignored.

    Returns:
        Tuple (speed: int | None, confidence: float).
        Returns (None, 0.0) if no valid speed is found.
    """
    best_speed = None
    best_conf = 0.0
    for (_, text, conf) in ocr_results:
        if conf < conf_threshold:
            continue
        # Strip any non-digit characters (e.g. 'MPH', stray punctuation,
        # OCR artefacts from the sign border).
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            continue
        try:
            value = int(digits)
        except ValueError:
            continue
        # Validate against the known-good speed set and keep only the best match.
        if value in _VALID_SPEEDS and conf > best_conf:
            best_speed = value
            best_conf = conf
    return best_speed, best_conf


class SpeedLimitOCRNode:
    """ROS node that reads speed-limit numerals from sign crops via EasyOCR.

    Subscribes to the sign classifier output and the matching source image,
    filters for speed-limit signs, runs the OCR preprocessing pipeline, and
    publishes SpeedLimitArray messages for the autonomy stack to consume.
    """

    def __init__(self):
        # ── ROS parameters ────────────────────────────────────────────────────
        # Pixels added around the bounding box before cropping.  10 px is wider
        # than the classifier's default (5 px) because OCR benefits from seeing
        # a bit of the sign frame as a spatial anchor.
        self.roi_padding = rospy.get_param("~roi_padding", 10)
        # Upscale crops shorter than this before passing to OCR.
        self.min_crop_height = rospy.get_param("~min_crop_height", 64)
        # Minimum EasyOCR word confidence to accept.
        self.conf_threshold = rospy.get_param("~ocr_confidence_threshold", 0.5)
        # GPU flag: True when device param is a digit (CUDA index), False for CPU.
        use_gpu = str(rospy.get_param("~device", "0")).isdigit()

        sign_topic   = rospy.get_param("~sign_topic",   "/perception/traffic_signs")
        img_topic    = rospy.get_param("~image_topic",  "/yolov10/image_raw")
        output_topic = rospy.get_param("~output_topic", "/perception/speed_limit")

        # ── EasyOCR initialisation ────────────────────────────────────────────
        # Creating the Reader loads CRAFT (text detector) and CRNN (recogniser)
        # weights.  This is intentionally done here, once, rather than inside the
        # callback to avoid ~2 s startup overhead on every invocation.
        # First run: downloads ~100 MB to ~/.EasyOCR/model/.
        import easyocr
        rospy.loginfo("Initializing EasyOCR reader (first run downloads ~100MB)...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        rospy.loginfo("EasyOCR ready")

        # ── Lazy message imports ──────────────────────────────────────────────
        from yolov10_ros.msg import ClassifiedSignArray, SpeedLimit, SpeedLimitArray
        self.SpeedLimit = SpeedLimit
        self.SpeedLimitArray = SpeedLimitArray

        # ── Synchronized subscribers ──────────────────────────────────────────
        # The sign classifier and image republisher both originate from the same
        # detector callback, so their timestamps should be nearly identical.
        # slop=0.1 s provides a generous tolerance for pipeline jitter.
        sign_sub = message_filters.Subscriber(sign_topic, ClassifiedSignArray)
        img_sub  = message_filters.Subscriber(img_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sign_sub, img_sub], queue_size=10, slop=0.1
        )
        sync.registerCallback(self.callback)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.pub = rospy.Publisher(output_topic, SpeedLimitArray, queue_size=10)
        rospy.loginfo("Speed limit OCR node ready")

    def callback(self, sign_msg, img_msg):
        """Process one synchronised (ClassifiedSignArray, Image) pair.

        For each speed_limit sign in the array:
          1. Crop the ROI from the source image (with padding).
          2. Preprocess for OCR (upscale + grayscale + CLAHE).
          3. Run EasyOCR with digit-only allowlist.
          4. Validate the parsed integer against the legal speed set.
          5. Append to the output message if valid.

        Publishes SpeedLimitArray only if at least one sign was parsed.

        Args:
            sign_msg: yolov10_ros/ClassifiedSignArray from the sign classifier.
            img_msg:  sensor_msgs/Image matching the sign_msg timestamp.
        """
        # Fast-path: most frames will not contain a speed limit sign.
        speed_signs = [s for s in sign_msg.signs if s.sign_type == "speed_limit"]
        if not speed_signs:
            return

        # Decode the image once; all crops in this frame share the same array.
        image = imgmsg_to_cv2(img_msg)
        out_msg = self.SpeedLimitArray()
        out_msg.header = sign_msg.header

        for sign in speed_signs:
            # Expand the box slightly — OCR benefits from seeing the surrounding
            # red border of the speed limit sign as a boundary reference.
            crop = crop_roi(image, sign.xmin, sign.ymin, sign.xmax, sign.ymax,
                            padding=self.roi_padding)
            if crop.size == 0:
                continue

            # Preprocess: upscale → grayscale → CLAHE
            processed = _preprocess_crop(crop, self.min_crop_height)

            # Run EasyOCR.  allowlist='0123456789' restricts the character set to
            # digits only, which eliminates most misreads.
            ocr_results = self.reader.readtext(
                processed, allowlist='0123456789', detail=1
            )

            speed, conf = _parse_speed(ocr_results, self.conf_threshold)
            if speed is None:
                # No valid speed found — log at DEBUG level (not WARN) to avoid
                # flooding the console; partial reads are expected on small crops.
                rospy.logdebug("OCR found no valid speed limit in crop")
                continue

            sl = self.SpeedLimit()
            sl.speed = speed
            sl.display = "speed limit: {}".format(speed)   # e.g. "speed limit: 35"
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
