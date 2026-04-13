"""
NFL Route Tracker - Tracking Module
====================================

Motion detection and trajectory tracking.
"""

from .trajectory import Detection, Trajectory, TrajectoryStore
from .detection_tracker import DetectionTracker
from .bytetrack_tracker import ByteTrackTracker, Track
from .trajectory_merger import TrajectoryMerger, merge_trajectory_store
from .field_camera_stabilizer import FieldCameraStabilizer
from .field_orientation_detector import FieldOrientationDetector, FieldOrientation

# Import DetectionTrackerConfig from core.config
from ..core.config import (DetectionTrackerConfig,
                           TrackerConfig,
                           CameraStabilizerConfig,
                           FieldCameraStabilizerConfig,
                           FieldOrientationConfig,
                           get_pipeline_config,
                           get_default_pipeline_config)

__all__ = [
    'Detection',
    'Trajectory',
    'TrajectoryStore',
    'Track',
    'ByteTrackTracker',
    'TrajectoryMerger',
    'merge_trajectory_store',
    'FieldCameraStabilizer',
    'FieldOrientationDetector',
    'FieldOrientation',

    'DetectionTracker',
    'DetectionTrackerConfig',
    'TrackerConfig',
    'CameraStabilizerConfig',
    'FieldCameraStabilizerConfig',
    'FieldOrientationConfig',

    # Config functions
    'get_pipeline_config',
    'get_default_pipeline_config',
]
