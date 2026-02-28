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
    """
    # test videos
    # threshold: int = 25
    # min_contour_area: int = 100
    # blur_kernel_size: Tuple[int, int] = (7, 7)
    # dilation_iterations: int = 2

    # nfl videos
    threshold: int = 40
    min_contour_area: int = 700
    blur_kernel_size: Tuple[int, int] = (17, 17)
    dilation_iterations: int = 2

    # adding bounding box filter dimensions (aspect ratio - width / height, size - area)
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 1.5
    min_area: int = 2500         
    max_area: int = 10000       

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure blur kernel is odd (OpenCV requirement)
        if self.blur_kernel_size[0] % 2 == 0 or self.blur_kernel_size[1] % 2 == 0:
            raise ValueError("Blur kernel dimensions must be odd numbers")

        # Ensure threshold is in valid range
        if not 0 <= self.threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255")

        # can probably delete, will output when initializing a motion tracker object
        # print(f"[CONFIG] MotionTrackerConfig initialized:")
        # print(f"Threshold: {self.threshold}")
        # print(f"Min contour area: {self.min_contour_area} pixels")
        # print(f"Blur kernel: {self.blur_kernel_size}")
        # print(f"Dilation iterations: {self.dilation_iterations}")


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