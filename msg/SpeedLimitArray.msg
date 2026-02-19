# SpeedLimitArray.msg
# --------------------
# All successfully parsed speed limit signs for a single camera frame.
#
# Published by the speed_limit_ocr_node on /perception/speed_limit.
# Only signs whose OCR result passes the confidence threshold AND whose
# numeric value is a valid US speed limit are included.
# If no signs pass, this message is not published.
#
# Autonomy stack consumers should use the most recent message from this topic
# to determine the currently posted speed limit.  The bounding box fields in
# each SpeedLimit can be used to associate the reading with a spatial location
# in the image if needed (e.g. for HUD overlay or sensor fusion).

Header       header        # Timestamp from the source ClassifiedSignArray
SpeedLimit[] speed_limits  # Successfully parsed speed limit signs
