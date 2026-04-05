"""
NFL Route Tracker - Tracking Module
====================================

Motion detection and trajectory tracking.
"""

from .trajectory import Detection, Trajectory, TrajectoryStore
from .detection_tracker import DetectionTracker
from .bytetrack_tracker import ByteTrackTracker, Track

# Import DetectionTrackerConfig from core.config
from ..core.config import (DetectionTrackerConfig,
                           TrackerConfig,
                           get_pipeline_config,
                           get_default_pipeline_config)

__all__ = [
    'Detection',
    'Trajectory',
    'TrajectoryStore',
    'Track',
    'ByteTrackTracker',

    'DetectionTracker',
    'DetectionTrackerConfig',
    'TrackerConfig',

    # Config functions
    'get_pipeline_config',
    'get_default_pipeline_config',
]
