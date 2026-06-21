# YOLOv3 Object Detection - PyTorch Implementation

PyTorch implementation of YOLOv3 for object detection with comparison to YOLOv8.

## Features

- YOLOv3 implementation with Darknet-53 backbone
- 3-scale detection (small, medium, large objects)
- Manual NMS implementation
- YOLOv8 comparison experiments
- Pre-trained weights support

## Project Structure

```
.
├── config.py              # YOLOv3 anchors configuration
├── model.py               # Neural network architecture
├── weights.py             # Weight loading utilities
├── utils.py               # Preprocessing and NMS functions
├── inference.py           # Prediction pipeline
├── main.py                # Main execution script
├── run_experiments.py     # Comparative experiments
├── generate_comparison_plots.py       # Visualization for standard images
├── generate_comparison_plots_hard.py  # Visualization for hard images
└── run_experiments_metrics.py         # Performance metrics
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- ultralytics (for YOLOv8 comparison)

## Installation

```bash
# Using UV (recommended)
uv sync

# Using Poetry
poetry install
```

## Quick Start

### Download Weights

```bash
wget https://pjreddie.com/media/files/yolov3.weights
mv yolov3.weights weights/
```

### Run Detection

```bash
# YOLOv3
uv run python main.py

# YOLOv8 (ultralytics)
# Edit main.py to uncomment main_ultralytics()
```

### Run Experiments

```bash
# Compare YOLOv3 vs YOLOv8
uv run python run_experiments.py

# Generate comparison plots
uv run python generate_comparison_plots.py
uv run python generate_comparison_plots_hard.py

# Performance metrics
uv run python run_experiments_metrics.py
```

## Configuration

Adjust parameters in `main.py`:
- `score_threshold`: Confidence threshold (default: 0.5)
- `iou_threshold`: IoU threshold for NMS (default: 0.4)

## Key Components

### Manual NMS
Custom Non-Maximum Suppression implementation in `utils.py`:
- Sorts boxes by confidence score
- Calculates IoU between boxes
- Suppresses overlapping detections per class

### YOLOv3 Architecture
- Darknet-53 backbone
- Feature Pyramid Network (FPN)
- 3 detection scales: 13×13, 26×26, 52×52

### Experiments
- Compares YOLOv3 vs YOLOv8 performance
- Tests different confidence/IoU thresholds
- Generates side-by-side visualizations

## Output

Results are saved in:
- `exps/exp_v3/` - YOLOv3 predictions
- `exps/exp_v26/` - YOLOv8 predictions
- `comparisons/` - Comparison visualizations

## References

- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [Darknet](https://pjreddie.com/darknet/yolo/)
- [COCO Dataset](https://cocodataset.org/)