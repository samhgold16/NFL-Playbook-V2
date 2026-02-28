"""
Running entire demo pipeline

From making test videos (or using real all-22 film) 
to processing the video to tracking movement to plotting
"""

import sys
from pathlib import Path

# Add src to path for development (before package is installed)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# import all packages wanting to test
from nfl_route_tracker import (
    # current packages, in order

    # testing class and global attributes
    TestVideoGenerator,
    VideoConfig, # from test_video_generator file

    # loading videos 
    VideoLoader,

    # identifying players
    MotionTrackerConfig, # from config.py file
    MotionTracker,

    # tracking identification over time
    TrajectoryStore,

    # rendering bounding boxes on original video
    TrajectoryVisualizer,
    create_tracking_video, # from visualizer.py
)