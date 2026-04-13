"""
NFL Route Tracker - Configuration Module
=========================================

This module contains all configuration settings and constants used throughout the package.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np

# =============================================================================
# NFL-SPECIFIC DETECTION FILTERING CONFIG
# =============================================================================

@dataclass
class NFLDetectionFilterConfig:
    """
    Configuration and tuning parameters for NFL-specific detection filtering.
    """
    # Area constraints (width * height in pixels)
    min_area: int = 150 # 2000 # 400
    max_area: int = 7500 # 20000

    # Aspect ratio constraints (width / height)
    min_aspect_ratio: float = 0.25 # .2
    max_aspect_ratio: float = 1.75 # 1.25

    # Confidence thresholds, ignore confidence below this and/or use ByteTrack association instead
    min_confidence: float = 0.15 # .15
    low_confidence: float = 0.05 # .05

    # Field zone thresholds (y-position in frame) for a 1920x984 video, with y = 984
    # All-22 camera angle: top = far, bottom = near
    near_y_threshold: int = 734
    far_y_threshold: int = 250

    # Near players (closer to camera)
    near_area_range: Tuple[int, int] = (350, 10000) # (2500, 20000) # (600, 20000)
    near_aspect_range: Tuple[float, float] = (0.2, 1.75) # (0.35, 0.85)

    # Far players (further from camera)
    far_area_range: Tuple[int, int] = (150, 7500) # (800, 8000) # (200, 20000))
    far_aspect_range: Tuple[float, float] = (0.2, 1.75) # (0.2, 0.6)

    # Mid-range players
    mid_area_range: Tuple[int, int] = (250, 7500) # (1500, 12000) # (400, 20000)
    mid_aspect_range: Tuple[float, float] = (0.25, 1.75) # (0.25, 0.7)

    # Vertical position constraints
    min_y_position: int = 10 # boundaries to not consider
    max_y_position: int = 974

    # Overlap merging
    merge_overlaps: bool = True
    merge_iou_threshold: float = 0.45 # .35 # CHANGE NEXT  # .8

    def __post_init__(self):
        """Validate configuration."""
        if self.min_area >= self.max_area:
            raise ValueError("min_area must be less than max_area")
        if self.min_aspect_ratio >= self.max_aspect_ratio:
            raise ValueError("min_aspect_ratio must be less than max_aspect_ratio")
        
# =============================================================================
#  CAMERA MOTION COMPENSATION CONFIG
# =============================================================================

@dataclass
class CameraStabilizerConfig:
    """
    Configuration for camera motion compensation.
    """
    enabled: bool = True # disable if using ByteTrack????
    feature_method: str = 'shi-tomasi'  # orb, sift, shi-tomasi for speed vs accuracy
    max_features: int = 400  # smaller is less robust, larger is slower (0, 1000)
    quality_level: float = 0.01 
    min_distance: float = 5.0
    ransac_threshold: float = 3.0 # (1, 10) strict consistent matches vs lenient more matches
    smoothing_window: int = 5 # frames to avg homography over for smoothing (1 = no smoothing, 20 = more smoothing)
    motion_threshold: float = 1.0  # Skip if motion < 1 pixel

    def __post_init__(self):
        """Validate configuration."""
        if self.feature_method not in ['orb', 'sift', 'shi-tomasi']:
            raise ValueError("feature_method must be 'orb', 'sift', or 'shi-tomasi'")
        if self.max_features < 10:
            raise ValueError("max_features must be at least 10")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")
        
@dataclass
class FieldOrientationConfig:
    """
    Configuration for field orientation detection and perspective correction.
    """
    enabled: bool = True  # Enable field orientation detection
    video_width: int = 1920  # Video frame width
    video_height: int = 984  # Video frame height

    # Edge detection parameters (for Hough line detection)
    canny_low: int = 50  # Lower Canny threshold
    canny_high: int = 150  # Upper Canny threshold

    # Hough line transform parameters
    hough_threshold: int = 100  # Minimum votes for line detection
    hough_min_line_length: int = 100  # Minimum line length in pixels
    hough_max_line_gap: int = 50  # Maximum gap between line segments

    # Line classification
    angle_tolerance: float = 15.0  # Degrees - lines within this of horizontal/vertical are field lines

    # Detection confidence
    min_field_lines: int = 3  # Minimum lines to detect for valid field orientation

    # Output scaling
    target_field_width: int = 1920  # Output width (pixels per 53.3 yards)
    target_field_height: int = 984  # Output height (pixels per 100 yards)

    def __post_init__(self):
        """Validate configuration."""
        if self.canny_low < 0 or self.canny_high < 0:
            raise ValueError("Canny thresholds must be non-negative")
        if self.canny_low >= self.canny_high:
            raise ValueError("canny_low must be less than canny_high")


# =============================================================================
# DETECTOR CONFIG (YOLO)
# =============================================================================

@dataclass
class DetectorConfig:
    """
    Configuration for PlayerDetector.
    """
    model_name: str = 'yolov8m.pt'  #yolov8s.pt, 8n, 
    confidence_threshold: float = 0.1 # .1
    # class 0 is associated to people
    classes: List[int] = field(default_factory = lambda: [0]) 
    device: str = 'auto'
    imgsz: int = 960 # 1280

    # sanity checks
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.imgsz % 32 != 0:
            raise ValueError("imgsz must be multiple of 32")
        

# =============================================================================
# TRACKER CONFIG (DeepSORT) --> NOW # BYTETRACK CONFIG
# =============================================================================

@dataclass
class TrackerConfig:
    """
    Configuration for DeepSORT tracker.
    # for now, replacing DeepSORT with ByteTrack
    """
    track_high_thresh: float = 0.4 # lower/lenient vs higher/strict detection filtering [0, 1]
    track_low_thresh: float = 0.05 # lower/lenient vs higher/strict detection filtering FOR SECOND PASS [0,1]
    new_track_thresh: float = 0.6 # threshold for creating new tracks [0,1]
    
    match_thresh: float = 0.8 # overlap to match detection to existing track or not [0,1]
    track_buffer: int = 90 # frames to keep lost track alive
    max_trajectory_gap: int = 50 # maximum gap (in frames) to allow for interpolation when a track is temporarily lost
    iou_threshold: float = 0.35    
    
    gmc_method: str = 'sift' # camera stabilization (can remove camera stabilizer now???)
    gmc_downscale: float = 2.0 # lower/slower/accurate vs higher/faster/less accurate [1.0, inf]
    
    min_trajectory_length: int = 10 # minimum number of frames for a valid trajectory

    # YOLO conf and ghost filtering
    confidence_threshold: float = 0.05
    filter_ghost_boxes: bool = True
    fuse_score: bool = True # whether to fuse detection confidence score with track confidence for filtering (ByteTrack specific)

    # Trajectory Post-Processing (trajectory merger)
    enable_trajectory_merging: bool = True  # Enable/disable trajectory merging
    merger_spatial_threshold: float = 150.0  # Max distance (pixels) between trajectory end and next start
    merger_temporal_threshold: int = 45  # Max frame gap to consider for merging (~0.75 seconds at 60fps)
    merger_confidence_threshold: float = 0.25  # Minimum score to perform merge
    merger_density_radius: float = 200.0  # Neighborhood radius (pixels) for crowd density check
    merger_density_threshold: int = 4  # Max endpoints in radius before region is considered crowded
    merger_max_merges: int = 2  # Maximum merges per trajectory to prevent chain merging

    def __post_init__(self):
        """Validate configuration."""
        valid_gmc = {'ecc', 'sift', 'orb', 'sparseOptFlow', None} # orb is fastest, ecc if most accurate
        if self.gmc_method not in valid_gmc:
            raise ValueError(f"gmc_method must be one of {valid_gmc}")
        valid_thresh = {'track_thresh': self.track_high_thresh, 
                        'track_low_thresh': self.track_low_thresh, 
                        'new_track_thresh': self.new_track_thresh, 
                        'match_thresh': self.match_thresh}
        if any(x < 0 for x in valid_thresh.values()):
            raise ValueError("All thresholds must be between 0 and 1")
        
# =============================================================================
# UNIFIED DETECTION + TRACKING PIPELINE CONFIG (DetectorConfig + TrackerConfig)
# =============================================================================

@dataclass
class DetectionTrackerConfig:
    """
    Configuration for the unified detection + tracking pipeline.
    """
    detector_config: Optional[DetectorConfig] = None
    tracker_config: Optional[TrackerConfig] = None
    nfl_filter_config: Optional[NFLDetectionFilterConfig] = None
    camera_config: Optional[CameraStabilizerConfig] = None
    field_orientation_config: Optional[FieldOrientationConfig] = None

    # output options
    verbose: bool = True
    progress_interval: int = 50
    save_video: bool = True  # Save annotated output video
    save_trajectories: bool = True  # Save trajectory JSON

    def __post_init__(self):
        """Initialize with defaults if not provided."""
        if self.detector_config is None:
            self.detector_config = DetectorConfig()
        if self.tracker_config is None:
            self.tracker_config = TrackerConfig()
        if self.nfl_filter_config is None:
            self.nfl_filter_config = NFLDetectionFilterConfig()
        if self.camera_config is None:
            self.camera_config = CameraStabilizerConfig()
        if self.field_orientation_config is None:
            self.field_orientation_config = FieldOrientationConfig()

# defining nfl field, used later
@dataclass
class NFLFieldConstants:
    """
    Official NFL field dimensions, for homography transformation, visualization scaling, coordinate validation
    """
    FIELD_LENGTH: float = 120.0
    FIELD_WIDTH: float = 53.33
    END_ZONE_DEPTH: float = 10.0
    PLAYING_FIELD_LENGTH: float = 100.0
    HASH_MARK_WIDTH: float = 18.5 / 3
    YARD_LINE_SPACING: float = 5.0
    GOAL_POST_HEIGHT: float = 10.0 / 3
    CROSSBAR_WIDTH: float = 18.5 / 3

# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

# Default configurations for quick access
DEFAULT_DETECTOR_CONFIG = DetectorConfig()
DEFAULT_TRACKER_CONFIG = TrackerConfig()
DEFAULT_NFL_FILTER_CONFIG = NFLDetectionFilterConfig()
NFL_FIELD = NFLFieldConstants()


# setting up overall pipeline config with all defaults, can be overridden by user when initializing pipeline
def get_pipeline_config() -> DetectionTrackerConfig:

    return DetectionTrackerConfig(detector_config = DetectorConfig(model_name = 'yolov8l.pt',
                                                                   confidence_threshold = 0.05,
                                                                   imgsz = 960), # 1280
                                  tracker_config = TrackerConfig(track_high_thresh = 0.5, track_low_thresh = 0.05,
                                                                new_track_thresh = 0.15, track_buffer = 90,
                                                                match_thresh = 0.65,
                                                                gmc_method = None,  # CameraStabilizer handles motion
                                                                gmc_downscale = 2.0, min_trajectory_length = 30,
                                                                max_trajectory_gap = 50, confidence_threshold = 0.05,
                                                                filter_ghost_boxes = False, fuse_score = True,
                                                                iou_threshold = 0.35,
                                                                enable_trajectory_merging = True,
                                                                merger_spatial_threshold = 150.0,
                                                                merger_temporal_threshold = 45,
                                                                merger_confidence_threshold = 0.5,
                                                                merger_density_radius = 200.0,
                                                                merger_density_threshold = 4,
                                                                merger_max_merges = 2),
                                  nfl_filter_config = NFLDetectionFilterConfig(merge_iou_threshold = 0.35, min_confidence = 0.15, low_confidence = 0.05),
                                  camera_config = CameraStabilizerConfig(enabled = True, feature_method = 'shi-tomasi', max_features = 400,
                                                                         ransac_threshold = 3.0, smoothing_window = 5, motion_threshold = 1.0),
                                  field_orientation_config = FieldOrientationConfig(enabled = True,
                                                                                    video_width = 1920,
                                                                                    video_height = 984,
                                                                                    canny_low = 50,
                                                                                    canny_high = 150,
                                                                                    hough_threshold = 100,
                                                                                    hough_min_line_length = 100,
                                                                                    hough_max_line_gap = 50,
                                                                                    angle_tolerance = 15.0,
                                                                                    min_field_lines = 3),
                                  verbose = True,
                                  progress_interval = 50,
                                  save_video = True,
                                  save_trajectories = True)

# Alias for backwards compatibility
def get_default_pipeline_config() -> DetectionTrackerConfig:
    """Alias for get_pipeline_config() for backwards compatibility."""
    return get_pipeline_config()