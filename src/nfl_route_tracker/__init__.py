"""
NFL Route Tracker
=================

A Python package for extracting player tracking data from NFL All-22 film
using computer vision techniques.
"""

__version__ = "0.2.0"
__author__ = "Sam Gold"

# Core components
from .core import (
    MotionTrackerConfig,
    NFLFieldConstants,
    DEFAULT_MOTION_CONFIG,
    NFL_FIELD,
    VideoLoader,
    VideoMetadata
)

# UNCOMMENT WHEN IMPLEMENT PHASE 2
# from .detection import (
#     PlayerDetector,
#     DetectionResult
# )

# Tracking components
from .tracking import (
    # Data structures (both phases)
    Detection,
    Trajectory,
    TrajectoryStore,

    # Phase 1: Motion-based
    MotionTracker,
    MotionBlob,
    draw_detections,

    # UNCOMMENT WHEN IMPLEMENT PHASE 2
    # ObjectTracker,
    # TrackerConfig,
    # Track,
    # DetectionTracker,
    # DetectionTrackerConfig,
)

# Visualization components
from .visualizations import (
    TrajectoryVisualizer,
    create_tracking_video
)

# Utility components
from .utils import (
    TestVideoGenerator,
    VideoConfig,
    MovingObject,
    # create_simple_test_video,
    # create_route_test_suite
)

__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Core
    'MotionTrackerConfig',
    'NFLFieldConstants',
    'DEFAULT_MOTION_CONFIG',
    'NFL_FIELD',
    'VideoLoader',
    'VideoMetadata',

    # Detection (Phase 2)
    # UNCOMMENT
    # 'PlayerDetector',
    # 'DetectionResult',

    # Tracking - Data structures
    'Detection',
    'Trajectory',
    'TrajectoryStore',

    # Tracking - Phase 1
    'MotionTracker',
    'MotionBlob',
    'draw_detections',

    # Tracking - Phase 2
    # UNCOMMENET
    # 'ObjectTracker',
    # 'TrackerConfig',
    # 'Track',
    # 'DetectionTracker',
    # 'DetectionTrackerConfig',

    # Visualization
    'TrajectoryVisualizer',
    'create_tracking_video',

    # Utils
    'TestVideoGenerator',
    'VideoConfig',
    'MovingObject',
    # 'create_simple_test_video',
    # 'create_route_test_suite',
]
