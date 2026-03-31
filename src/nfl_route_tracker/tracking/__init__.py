"""
NFL Route Tracker - Tracking Module
====================================

Motion detection and trajectory tracking.
"""

from .trajectory import Detection, Trajectory, TrajectoryStore
from .object_tracker import ObjectTracker, Track, TrackerConfig
from .detection_tracker import DetectionTracker

# Import DetectionTrackerConfig from core.config
from ..core.config import DetectionTrackerConfig

__all__ = [
    'Detection',
    'Trajectory',
    'TrajectoryStore',
    'ObjectTracker',
    'Track',
    'TrackerConfig',
    'DetectionTracker',
    'DetectionTrackerConfig',
]
