# ClassifiedSignArray.msg
# ------------------------
# All classified traffic signs for a single camera frame.
#
# Published by the sign_classifier_node on /perception/traffic_signs.
# Only signs that pass the confidence threshold are included.
# If no signs pass, this message is not published.
#
# The SpeedLimitOCRNode subscribes to this topic and filters for
# signs where sign_type == "speed_limit" to run numeric OCR.

Header           header  # Timestamp from the source DetectionArray
ClassifiedSign[] signs   # Confirmed sign classifications for this frame
