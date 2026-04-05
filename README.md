# NFL Motion Tracker and Route Classifier

## CURRENT PROGRESS

*Version 1 - Temporal Differencing*

![tracked_output-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/1ac16c65-2a76-4c6d-adad-869677f42cd5)

*Version 2 - YOLO + DeepSORT*

![test1_tracked-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/787da9a5-ceaf-4a63-b80a-ac6d4d74b437)

## Installation Steps

*INPUT HERE*

## Simple Usage

*INPUT HERE*

## Project Structure

```
src/nfl_route_tracker/
├── core/
│   ├── __init__.py
│   ├── config.py               # ALL hyperparameters (ByteTrack, YOLO, NFL filter)
│   └── video_loader.py         # CV2 video I/O wrapper
│
├── detection/
│   ├── __init__.py
│   ├── player_detector.py      # YOLO wrapper (DetectionResult)
│   └── nfl_filter.py           # NFL-specific filtering
│
├── tracking/
│   ├── __init__.py
│   ├── bytetrack_tracker.py    # ByteTrack tracker
│   ├── detection_tracker.py    # Main pipeline orchestrator
│   ├── trajectory.py           # Data structures (Detection, Trajectory, TrajectoryStore)
│   ├── camera_stabilizer.py    # Properly mapping trajectories
│   └── bytetrack.yaml          # Used for ByteTrack
│
└── visualizations/
    ├── __init__.py
    └── visualizer.py           # Matplotlib trajectory plotting
```

## Individual Packages

*INSERT HERE*

### VideoLoader

Example usage:

```python
with VideoLoader("video.mp4") as video:
    # See video properties
    print(video.metadata)
    # iterate through each frame in video
    for frame_num, frame in video:
        process_frame(frame)
        pass
```

## Customizeable Configuration

*INSERT HERE*

## Hyperparameter Definitions

### Detector Class (YOLO)

| Parameter | Options | Description |
|---|---|---|
| model_name | yolov8m.pt, s, n  | YOLO model size for nano, small, medium, etc. Tradeoff between speed and accuracy. |
| confidence_threshold | [0, 10]| Minimum YOLO confidence to accept a detection |
| imgsz | 32, 64, ..., 960, 1290 | Input image resolution |

### Tracker Class (DeepSORT)

| Parameter | Options | Description |
|---|---|---|
| max_age | [0, ...]| Frames to keep tracking a player after last detection |
| n_init | 1, 2, 3 | Confirmations needed before showing track |
| max_iou_distance | [0, 1] | Threshold for BB overlap matching (smaller is more aggresive, larger is more lenient) |
| max_cosine_distance | [0, 2] | Threshold for BB appearence matching  (smaller is more aggresive, larger is more lenient) |
| min_hits | 1 | Minimum detections before track appears |

### NFL Detection Filter

| Parameter | Options | Description |
|---|---|---|
| min_area | 100+ | Minimum pixel area for a valid BB |
| max_area | 30000- | Minimum pixel area for a valid BB |
| min_aspect_ratio | 0+ | Minimum width/height ratio of BB |
| max_aspect_ratio | 2- | Maximum width/height ratio of BB |
| near_y_threshold | 700+ | Y Position marking field that is far away (top of video) |
| far_y_threshold| 200- | Y Position marking field that is close (bottom of video)  |
| merge_iou_threshold | [0, 1] | Threshold above which nearby detections are merged |

** more to add, check config.py file**

### Camera Stablization

| Parameter | Options | Description |
|---|---|---|
| feature_method | shi-tosami, orb, sift | Feature detection algorithm |
| max_features | [0, 1000+]| Max features to track  |
| smoothing_window | [0, 10+] | Frames to average for smoothing |
| ransac_threshold | [0, 10] | Outlier threshold |
| motion_threshold | [0, 10]| Minimum motion to apply correct |

### Personal Notes 

*NEED TO FILL OUT PROPERLY*

To run code, run `python demo_code/demo.py` from root directory.
