"""
NFL Route Tracker - Configuration Module
=========================================

This module contains all configuration settings and constants used throughout
the package. By centralizing configuration here, we can easily adjust parameters
without hunting through multiple files.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from dataclasses import dataclass, field
from typing import List

# phase 1 global variables
@dataclass
class MotionTrackerConfig:
    """
    Configuration for the MotionTracker class.

    Attributes:
    -----------
    threshold : Pixel intensity difference required to be considered "motion" (0-255).
    min_contour_area : Minimum pixel area for a detected region to be considered a "moving object".
    blur_kernel_size : Size of Gaussian blur kernel applied before differencing (odd numbers)
    dilation_iterations :   Number of times to apply morphological dilation.
    max_tracking_distance : How easy it is for movement to be "remembered" for a given trajectory
    """
    # better settings for simple test videos
    # threshold: int = 25
    # min_contour_area: int = 100
    # blur_kernel_size: Tuple[int, int] = (7, 7)
    # dilation_iterations: int = 2
    # max_tracking_distance: float = 50.0

    # 'tuned' settings for nfl videos
    threshold: int = 40
    min_contour_area: int = 900
    blur_kernel_size: Tuple[int, int] = (15, 15)
    dilation_iterations: int = 2
    # to help with "object permamence" for associating blobs to trajectories
    max_tracking_distance: float = 250.0

    # adding bounding box filter dimensions (aspect ratio - width / height, size - area)
    min_aspect_ratio: float = 0.5
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


# phase 2 global variables
# same set of attributes functioning to how config.py works
@dataclass
class DetectorConfig:
    """
    Configuration for PlayerDetector.

    Attributes:
    -----------
    model_name : YOLO model type (...8n/8s/8m/8l/8x)
    confidence_threshold : Confidence need to accept detection
    classes : COCO class IDs to detect. Default [0] = person only.
    device : Device to run on: 'auto', 'cpu', 'cuda', 'mps' 
    imgsz : Input image size for YOLO (multiple of 32)

    """
    model_name: str = 'yolov8n.pt'  
    confidence_threshold: float = 0.2
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

# defining nfl field, used later
@dataclass
class NFLFieldConstants:
    """
    Official NFL field dimensions.

    These constants are used for:
    1. Homography transformation (Phase 5)
    2. Coordinate validation
    3. Visualization scaling

    All measurements in YARDS unless otherwise specified.

    """
    # Field dimensions
    FIELD_LENGTH: float = 120.0  # Including both end zones
    FIELD_WIDTH: float = 53.33   # 160 feet = 53.33 yards

    # End zones
    END_ZONE_DEPTH: float = 10.0

    # Playing field (excluding end zones)
    PLAYING_FIELD_LENGTH: float = 100.0

    # Hash marks (distance from sideline)
    HASH_MARK_WIDTH: float = 18.5 / 3  # 18.5 feet = 6.17 yards from center

    # Yard line spacing
    YARD_LINE_SPACING: float = 5.0  # Major lines every 5 yards

    # Goal post dimensions (for future reference)
    GOAL_POST_HEIGHT: float = 10.0 / 3  # 10 feet = 3.33 yards
    CROSSBAR_WIDTH: float = 18.5 / 3    # Same as hash mark width

# Default configurations 
DEFAULT_MOTION_CONFIG = MotionTrackerConfig()
NFL_FIELD = NFLFieldConstants()