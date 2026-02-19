# SpeedLimit.msg
# ---------------
# The parsed numeric value from a speed limit sign, produced by the
# speed_limit_ocr_node via EasyOCR.
#
# Published as part of SpeedLimitArray on /perception/speed_limit.
#
# The speed value is validated against the set of legal US posted speed limits
# (mph) before being published: {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
# 70, 75, 80, 85}.  Any OCR result outside this set is discarded.
#
# When OCR fails to produce a valid reading (e.g. the sign is too small,
# occluded, or at an angle), no SpeedLimit message is published for that sign —
# the sign classifier still publishes the ClassifiedSign so consumers know a
# speed-limit sign is present even without a confirmed value.

int32   speed           # Parsed speed limit in mph, e.g. 35
string  display         # Human-readable label, e.g. "speed limit: 35"
float64 ocr_confidence  # EasyOCR word confidence [0.0, 1.0]
int32   xmin            # Left edge of sign bounding box (pixels, original frame)
int32   ymin            # Top edge of sign bounding box (pixels, original frame)
int32   xmax            # Right edge of sign bounding box (pixels, original frame)
int32   ymax            # Bottom edge of sign bounding box (pixels, original frame)
