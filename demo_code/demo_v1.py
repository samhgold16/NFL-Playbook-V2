"""
NFL Route Tracker - Complete Pipeline Demo
Simple end-to-end test using trial_vid.mp4
"""
# importing all important packages
import sys
import os
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_route_tracker.core.config import MotionTrackerConfig
from nfl_route_tracker.tracking.motion_tracker import MotionTracker
from nfl_route_tracker.visualizations.visualizer import TrajectoryVisualizer, create_tracking_video

# helper formatting prints
def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_pipeline():
    """Run complete demonstration using specified video"""
    
    print_header("NFL Route Tracker - Phase 1 Demo")

    # paths 
    test_folder = Path(__file__).parent.parent / "data" / "video_test"
    # change video here for testing
    video_path = test_folder / "trial_vid2.mp4"
    output_folder = test_folder.parent / "viz_output"
    output_folder.mkdir(exist_ok = True)
    
    print(f"Input Video: {video_path}")
    print(f"Output Directory: {output_folder}\n")
    
    print_header("Step 1: Processing and Tracking Video")

    config = MotionTrackerConfig()
    print(f"Configuration:")
    print(f"  - Threshold: {config.threshold}")
    print(f"  - Min contour area: {config.min_contour_area} pixels")
    print(f"  - Blur kernel: {config.blur_kernel_size}")
    print(f"  - Dilation iterations: {config.dilation_iterations}\n")

    tracker = MotionTracker(config)
    store = tracker.process_video(str(video_path))
    print(f"Found {store.num_trajectories} trajectories\n")
    
    # Step 2: Create visualizations
    print_header("Step 2: Creating Visualizations")

    viz = TrajectoryVisualizer(figsize = (12, 8), style = 'dark_background')
    
    traj_plot = output_folder / "trajectories.png"
    viz.plot_trajectories(store, str(traj_plot), title = "Tracked Trajectories")
    print(f"Saved: {traj_plot.name}")

    
    # Step 3: Create annotated video (your key output!)
    print_header("Step 3: Adding Bounding Boxes")

    tracked_video = output_folder / "tracked_output.mp4"
    create_tracking_video(str(video_path), store, str(tracked_video))
    print(f"Saved: {tracked_video.name}")
    
    print("\nPIPELINE COMPLETE!")

if __name__ == "__main__":
    run_pipeline()