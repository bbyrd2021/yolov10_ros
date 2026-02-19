# DetectionArray.msg
# ------------------
# All object detections for a single camera frame, published by the
# YOLOv10 detector node on /yolov10/detections.
#
# The header timestamp matches the source camera frame timestamp so that
# message_filters.ApproximateTimeSynchronizer can pair this message with the
# corresponding Image on /yolov10/image_raw for ROI cropping in downstream
# classifier nodes.
#
# An empty detections array is published on frames with no detections above
# the confidence threshold.

Header      header      # Timestamp copied from the source camera message
Detection[] detections  # Zero or more detections; empty if nothing detected
