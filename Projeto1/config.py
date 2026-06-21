"""YOLOv3 configuration and constants."""

# YOLOv3 anchor boxes for 3 detection scales
YOLOV3_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # Scale 1 (13x13) - Large objects
    [(30, 61), (62, 45), (59, 119)],      # Scale 2 (26x26) - Medium objects
    [(10, 13), (16, 30), (33, 23)]        # Scale 3 (52x52) - Small objects
]
