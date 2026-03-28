"""
NFL Route Tracker Version 2- Complete Pipeline Demo
Simple end-to-end test using trial_vid.mp4, incorporating YOLO and DeepSort
"""

# importing all important packages
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_route_tracker.core.video_loader import VideoLoader
from nfl_route_tracker.tracking.trajectory import TrajectoryStore, Trajectory, Detection
from nfl_route_tracker.core.config import TrackerConfig, DetectorConfig, DetectionTrackerConfig
from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.tracking.object_tracker import ObjectTracker, Track
from nfl_route_tracker.tracking.detection_tracker import DetectionTracker
from nfl_route_tracker.visualizations.visualizer import TrajectoryVisualizer

# helper formatting prints
def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

# plotting function
def run_pipeline():
    """
    Run the complete Phase 2 demonstration.
    """
    
    print_header("NFL Route Tracker - Phase 2 Demo")

    # paths 
    test_folder = Path(__file__).parent.parent / "data" / "video_test"
    # change video here for testing
    num_vid = "1"
    video_path = test_folder / f"test{num_vid}.mp4"
    output_folder = test_folder.parent / "viz_output"
    output_folder.mkdir(exist_ok = True)
    output_video_path = output_folder / f"test{num_vid}_output.mp4"

    print(f"Input Video: {video_path}")
    print(f"Output Directory: {output_folder}\n")

    # setting up configurations here, or use default ones
    # to change, change config.py file
    config_detector = DetectorConfig()
    config_tracker = TrackerConfig()
    config_joint = DetectionTrackerConfig(detector_config = config_detector, 
                                          tracker_config = config_tracker)
    
    # passing joint attributes into main class
    pipeline = DetectionTracker(config_joint)
    print("Pipeline configuration:")
    print(f"  - YOLO model: {config_detector.model_name}")
    print(f"  - Confidence threshold: {config_detector.confidence_threshold}")
    print(f"  - DeepSORT max_age: {config_tracker.max_age}")
    print(f"  - DeepSORT n_init: {config_tracker.n_init}")
    print(f"  - IOU matching threshold: {config_tracker.max_iou_distance}")
    print(f"  - Appearance matching threshold: {config_tracker.max_cosine_distance}")
    print(f"  - Filtering enabled: {config_joint.enable_filtering}")
    print(f"  - Min area: {config_joint.min_area}")
    print(f"  - Max area: {config_joint.max_area}")
    print(f"  - NMS threshold: {config_joint.nms_threshold}")

    print_header("Step 2: Processing and Tracking Video")
    
    # process video
    traj_store = pipeline.process_video(str(video_path), output_video_path = str(output_video_path))

    print(f"Analyzing complete! Annotated video saved to {output_video_path}")

    print_header("Step 3: Analyze Trajectories")
    
    # getting stored trajectories from processed video
    trajectories = traj_store.get_all_trajectories()
    print(f"Total unique tracks: {len(trajectories)}")

    if len(trajectories) > 0:
        print_header("Step 4: Visualize Trajectories")

        # initializing plotter and plotting all trajectories
        viz = TrajectoryVisualizer(figsize = (12, 8))
        traj_plot_path = output_folder / "all_trajectories.png"
        viz.plot_trajectories(traj_store, output_path = str(traj_plot_path))
        print(f"Saved trajectory plot to: {traj_plot_path}")

    print("\nPIPELINE COMPLETE!")

# to test file
if __name__ == "__main__":
    run_pipeline()