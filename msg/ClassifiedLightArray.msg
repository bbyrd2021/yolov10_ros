# ClassifiedLightArray.msg
# -------------------------
# All classified traffic lights for a single camera frame.
#
# Published by the light_classifier_node on /perception/traffic_lights.
# Only lights that pass the confidence threshold (~confidence_threshold param)
# are included — low-confidence predictions are silently dropped.
# If no lights pass the threshold this message is not published at all.
#
# The header timestamp is propagated from the original DetectionArray so that
# downstream nodes can time-synchronise with the raw image if needed.

Header            header  # Timestamp from the source DetectionArray
ClassifiedLight[] lights  # Confirmed traffic-light states for this frame
