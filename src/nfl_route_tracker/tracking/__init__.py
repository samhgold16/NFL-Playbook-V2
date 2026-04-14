"""
NFL Route Tracker - Tracking Module
====================================

Motion detection and trajectory tracking.
"""

from .trajectory import Detection, Trajectory, TrajectoryStore
from .detection_tracker import DetectionTracker
from .bytetrack_tracker import ByteTrackTracker, Track
from .trajectory_merger import TrajectoryMerger, merge_trajectory_store
#from .field_orientation_detector import FieldOrientationDetector, FieldOrientation
from .fixed_field_orientation_detector import FixedFieldOrientationDetector, FixedFieldOrientation
from .final_field_transform import FinalFieldTransform, create_final_transform

# Import DetectionTrackerConfig from core.config
from ..core.config import (DetectionTrackerConfig,
                           TrackerConfig,
                           CameraStabilizerConfig,
                           FieldOrientationConfig,
                           get_pipeline_config,
                           get_default_pipeline_config)

__all__ = [
    # Data structures
    'Detection',
    'Trajectory',
    'TrajectoryStore',
    'Track',

    # Trackers
    'ByteTrackTracker',
    'DetectionTracker',

    # Stabilization/Orientation
    #'FieldOrientationDetector',
    #'FieldOrientation',
    'FixedFieldOrientationDetector',
    'FixedFieldOrientation',
    'FinalFieldTransform',
    'create_final_transform',

    # Trajectory processing
    'TrajectoryMerger',
    'merge_trajectory_store',

    # Config classes
    'DetectionTrackerConfig',
    'TrackerConfig',
    'CameraStabilizerConfig',
    'FieldOrientationConfig',

    # Config functions
    'get_pipeline_config',
    'get_default_pipeline_config',
]
