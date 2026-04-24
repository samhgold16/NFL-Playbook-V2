#!/usr/bin/env python3
"""
NFL Route Tracker - Final Corrected Field Transform
====================================================

This module handles the transformation from video coordinates to field coordinatesfor All-22 footage with slanted yard lines.
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FinalFieldTransform:
    """
    Final field transformation for All-22 footage with slanted yard lines.
    """

    def __init__(self, yard_line_angle: float, video_width: float, video_height: float):
        self.yard_line_angle = yard_line_angle
        self.video_width = video_width
        self.video_height = video_height
        self.center_x = video_width / 2
        self.center_y = video_height / 2

        # Pre-compute rotation matrix for -yard_line_angle for negative angle
        angle_rad = np.radians(-yard_line_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Centered rotation matrix
        self.rotation_matrix = np.array([[cos_a, -sin_a, self.center_x - cos_a * self.center_x + sin_a * self.center_y],
                                         [sin_a, cos_a, self.center_y - sin_a * self.center_x - cos_a * self.center_y],
                                          [0, 0, 1]])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform video coordinates to field coordinates.
        """
        point = np.array([x, y, 1.0])
        transformed = self.rotation_matrix @ point

        return (float(transformed[0]), float(transformed[1]))

    def get_field_depth(self, x: float, y: float) -> float:
        """
        Get the field depth position (Y on field, toward endzone).
        """
        _, field_y = self.transform_point(x, y)
        return field_y

    def get_field_width(self, x: float, y: float) -> float:
        """
        Get the field width position (sideline position).
        """
        field_x, _ = self.transform_point(x, y)
        return field_x

def create_final_transform(yard_line_angle: float, video_width: float, video_height: float) -> FinalFieldTransform:
    """Factory function to create final transform."""
    return FinalFieldTransform(yard_line_angle = yard_line_angle,
                              video_width = video_width,
                              video_height = video_height)
