# Detection.msg
# -------------
# A single object detection produced by the YOLOv10 detector node.
#
# Published as part of DetectionArray on /yolov10/detections.
# Downstream classifier nodes filter this message by class_name to find
# objects relevant to their model (e.g. "traffic light", "traffic sign").
#
# Bounding box coordinates are in the original camera image pixel space.
# No normalisation or scaling is applied — coordinates are directly usable
# for cropping via cv2/numpy array indexing.

string  class_name    # BDD100K class label, e.g. "car", "traffic light"
float64 confidence    # Detector softmax confidence [0.0, 1.0]
int32   xmin          # Left edge of bounding box (pixels)
int32   ymin          # Top edge of bounding box (pixels)
int32   xmax          # Right edge of bounding box (pixels)
int32   ymax          # Bottom edge of bounding box (pixels)
