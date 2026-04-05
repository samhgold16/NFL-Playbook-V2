"""
NFL Route Tracker
=================

A Python package for extracting player tracking data from NFL All-22 film
using computer vision techniques.
"""

__version__ = "0.4.0"
__author__ = "Sam Gold"

# Core components
from .core import (
    NFLFieldConstants,
    NFL_FIELD,
    VideoLoader,
    VideoMetadata
)

# Detection components
from .detection import (
    PlayerDetector,
    DetectionResult
)

# Tracking components
from .tracking import (
    # Data structures
    Detection,
    Trajectory,
    TrajectoryStore,

    # Tracker
    # ObjectTracker,
    ByteTrackTracker,
    TrackerConfig,
    Track,
    DetectionTracker,
    DetectionTrackerConfig,
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
    'NFLFieldConstants',
    'NFL_FIELD',
    'VideoLoader',
    'VideoMetadata',

    # Detection
    'PlayerDetector',
    'DetectionResult',

    # Tracking - Data structures
    'Detection',
    'Trajectory',
    'TrajectoryStore',

    # Tracking - Tracker
    #'ObjectTracker',
    'ByteTrackTracker',
    'TrackerConfig',
    'Track',
    'DetectionTracker',
    'DetectionTrackerConfig',

    # Visualization
    'TrajectoryVisualizer',
    'create_tracking_video',
]
