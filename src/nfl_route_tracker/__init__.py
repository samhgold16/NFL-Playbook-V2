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
    # 'create_simple_test_video',
    # 'create_route_test_suite',
]
