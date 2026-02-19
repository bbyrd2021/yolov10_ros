# ClassifiedLight.msg
# --------------------
# The classified state of a single traffic light detected in a camera frame.
#
# Published as part of ClassifiedLightArray on /perception/traffic_lights.
# Produced by the light_classifier_node after running an EfficientNet B0
# model on the cropped bounding box from the detector.
#
# State vocabulary (6 classes, trained with ImageFolder alphabetical order):
#   green_light   — Solid green (go)
#   green_left    — Green left-arrow (left turn permitted)
#   red_light     — Solid red (stop)
#   red_left      — Red left-arrow (left turn prohibited)
#   yellow_light  — Solid yellow (caution / prepare to stop)
#   yellow_left   — Yellow left-arrow (caution / left turn clearing)

string  state             # Predicted light state (see vocabulary above)
float64 state_confidence  # EfficientNet softmax probability [0.0, 1.0]
int32   xmin              # Left edge of bounding box (pixels, original frame)
int32   ymin              # Top edge of bounding box (pixels, original frame)
int32   xmax              # Right edge of bounding box (pixels, original frame)
int32   ymax              # Bottom edge of bounding box (pixels, original frame)
