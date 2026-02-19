# ClassifiedSign.msg
# -------------------
# The classified type of a single traffic sign detected in a camera frame.
#
# Published as part of ClassifiedSignArray on /perception/traffic_signs.
# Produced by the sign_classifier_node after running an EfficientNet B0
# model on the cropped bounding box from the detector.
#
# Sign type vocabulary (15 classes, US traffic signs):
#   detour               — Temporary detour
#   do_not_enter         — Wrong-way / do not enter
#   go_straight_only     — Mandatory straight ahead
#   no_left_turn         — No left turn prohibition
#   no_right_turn        — No right turn prohibition
#   no_straight          — No straight ahead
#   no_u_turn            — No U-turn
#   pedestrian_crossing  — Pedestrian crosswalk warning
#   railroad_crossing    — Railroad grade crossing
#   roadwork             — Construction / road work ahead
#   speed_limit          — Speed limit sign (numeric value read by OCR node)
#   stop                 — Stop sign
#   turn_left_only       — Mandatory left turn
#   turn_right_only      — Mandatory right turn
#   yield                — Yield / give way

string  sign_type         # Predicted sign category (see vocabulary above)
float64 type_confidence   # EfficientNet softmax probability [0.0, 1.0]
int32   xmin              # Left edge of bounding box (pixels, original frame)
int32   ymin              # Top edge of bounding box (pixels, original frame)
int32   xmax              # Right edge of bounding box (pixels, original frame)
int32   ymax              # Bottom edge of bounding box (pixels, original frame)
