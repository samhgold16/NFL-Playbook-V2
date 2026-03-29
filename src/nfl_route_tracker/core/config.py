"""
NFL Route Tracker - Configuration Module
=========================================

This module contains all configuration settings and constants used throughout the package.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# NFL-SPECIFIC DETECTION FILTERING CONFIG
# =============================================================================

@dataclass
class NFLDetectionFilterConfig:
    """
    Configuration and tuning parameters for NFL-specific detection filtering.
    """
    # Area constraints (width * height in pixels)
    min_area: int = 500 # 2000
    max_area: int = 17500 # 20000

    # Aspect ratio constraints (width / height)
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 1.15 # 1.25

    # Confidence thresholds, ignore confidence below this and/or use ByteTrack association instead
    min_confidence: float = 0.25
    low_confidence: float = 0.15

    # Field zone thresholds (y-position in frame)
    # All-22 camera angle: top = far, bottom = near
    near_y_threshold: int = 700
    far_y_threshold: int = 300

    # Near players (closer to camera)
    near_area_range: Tuple[int, int] = (2500, 20000)
    near_aspect_range: Tuple[float, float] = (0.35, 0.85)

    # Far players (further from camera)
    far_area_range: Tuple[int, int] = (800, 8000)
    far_aspect_range: Tuple[float, float] = (0.2, 0.6)

    # Mid-range players
    mid_area_range: Tuple[int, int] = (1500, 12000)
    mid_aspect_range: Tuple[float, float] = (0.25, 0.7)

    # Vertical position constraints
    min_y_position: int = 50
    max_y_position: int = 950

    # Overlap merging
    merge_overlaps: bool = True
    merge_iou_threshold: float = 0.35

    def __post_init__(self):
        """Validate configuration."""
        if self.min_area >= self.max_area:
            raise ValueError("min_area must be less than max_area")
        if self.min_aspect_ratio >= self.max_aspect_ratio:
            raise ValueError("min_aspect_ratio must be less than max_aspect_ratio")
        
# =============================================================================
# TEMPORAL AGGREGATION CONFIG
# =============================================================================
@dataclass
class TemporalAggregatorConfig:
    """
    Configuration for temporal detection aggregation, aggregating detections across multiple frames to improve stability
    """
    enabled: bool = True
    window_size: int = 6 # Number of frames to aggregate # 2
    stride: int = 2  # Step size between windows (1 = every frame) # 1
    aggregation_method: str = 'mean'  # 'mean', 'max', 'weighted'
    confidence_weight: float = 0.6  # Weight for confidence in weighted aggregation
    min_detection_count: int = 2  # Min frames a detection must appear in
    position_threshold: float = 30.0  # Max distance to consider same detection

    def __post_init__(self):
        """Validate configuration."""
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.aggregation_method not in ['mean', 'max', 'weighted']:
            raise ValueError("aggregation_method must be 'mean', 'max', or 'weighted'")

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
    imgsz: int = 1280

    # sanity checks
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.imgsz % 32 != 0:
            raise ValueError("imgsz must be multiple of 32")
        

# =============================================================================
# TRACKER CONFIG (DeepSORT)
# =============================================================================

@dataclass
class TrackerConfig:
    """
    Configuration for DeepSORT tracker.
    """
    max_age: int = 50 # frames to keep lost tracks alive    # 30
    n_init: int = 2 # frames to confirm a track before outputting    # 1 
    max_iou_distance: float = 0.2 # lower = more aggressive matching, higher = more lenient matching   # .15
    max_cosine_distance: float = 0.8 # lower = more aggressive matching, higher = more lenient matching   # .15
    nn_budget: int = 1000 # max number of features to store for each track (for appearance matching) # 1000
    embedder: str = 'mobilenet' 
    filter_ghost_boxes: bool = True
    min_hits: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.max_iou_distance <= 1.0:
            raise ValueError("max_iou_distance must be between 0 and 1")
        if not 0.0 <= self.max_cosine_distance <= 2.0:
            raise ValueError("max_cosine_distance must be between 0 and 2")


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
    temporal_config: Optional[TemporalAggregatorConfig] = None

    # output options
    verbose: bool = True
    progress_interval: int = 50
    save_video: bool = True  # Save annotated output video
    save_trajectories: bool = True  # Save trajectory JSON

    # Legacy filtering options (used if nfl_filter_config is None)
    enable_legacy_filtering: bool = False
    min_area: int = 2500
    max_area: int = 10000
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 1.25
    nms_threshold: float = 0.3

    def __post_init__(self):
        """Initialize with defaults if not provided."""
        if self.detector_config is None:
            self.detector_config = DetectorConfig()
        if self.tracker_config is None:
            self.tracker_config = TrackerConfig()
        if self.nfl_filter_config is None:
            self.nfl_filter_config = NFLDetectionFilterConfig()
        if self.temporal_config is None:
            self.temporal_config = TemporalAggregatorConfig()

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
DEFAULT_TEMPORAL_CONFIG = TemporalAggregatorConfig()
NFL_FIELD = NFLFieldConstants()


# setting up overall pipeline config with all defaults, can be overridden by user when initializing pipeline
def get_default_pipeline_config() -> DetectionTrackerConfig:

    return DetectionTrackerConfig(detector_config = DetectorConfig(model_name = 'yolov8m.pt',
                                                                   confidence_threshold = 0.25,
                                                                   imgsz = 1280),
                                  tracker_config = TrackerConfig(max_age = 50, n_init = 2,
                                                                 max_iou_distance = 0.8, max_cosine_distance = .4, 
                                                                 embedder = 'mobilenet'),
                                  nfl_filter_config = NFLDetectionFilterConfig(),
                                  temporal_config = TemporalAggregatorConfig(enabled = True, # False ???
                                                                             window_size = 2,
                                                                             aggregation_method = 'max'),
                                  verbose = True,
                                  progress_interval = 50,
                                  save_video = True,
                                  save_trajectories = True)

# =============================================================================
# UNUSED/OLD CODE
# =============================================================================

# phase 1 global variables
# not used anymore, can delete?
@dataclass
class MotionTrackerConfig:
    """
    Configuration for the MotionTracker class.
    *** no longer used ***
    """

    # 'tuned' settings for nfl videos
    threshold: int = 40
    min_contour_area: int = 900
    blur_kernel_size: Tuple[int, int] = (15, 15)
    dilation_iterations: int = 2
    max_tracking_distance: float = 250.0
    max_aspect_ratio: float = 1.5
    min_area: int = 2500         
    max_area: int = 12500       

    # ensuring inputs are within allowed boudns
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure blur kernel is odd (OpenCV requirement)
        if self.blur_kernel_size[0] % 2 == 0 or self.blur_kernel_size[1] % 2 == 0:
            raise ValueError("Blur kernel dimensions must be odd numbers")
        # Ensure threshold is in valid range
        if not 0 <= self.threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255")