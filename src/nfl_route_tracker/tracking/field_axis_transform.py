#!/usr/bin/env python3
"""
NFL Route Tracker - Direct Field-Axis Coordinate Transformation
===============================================================

This module provides a direct transformation that projects video coordinates
onto a top-down field coordinate system.

THE PROBLEM:
When yardlines are slanted in the video, a simple rotation doesn't preserve
spatial relationships. Players on the same yardline should have the same
X-coordinate after transformation, but pure rotation doesn't guarantee this.

THE SOLUTION:
Instead of rotation around a center point, we use PROJECTION:
1. Detect the angle of the yard lines (field X-axis)
2. Project each point onto an axis perpendicular to yard lines (true X)
3. Project each point onto an axis parallel to yard lines (true Y)

This ensures players lined up on the same yardline will have identical X values.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.tracking.field_orientation_detector import FieldOrientationDetector


class FieldAxisProjectionTransform:
    """
    Transforms video coordinates to field coordinates using axis projection.

    This is fundamentally different from rotation:
    - Rotation: Rotates all points by same angle around center
    - Projection: Projects each point onto the correct field axes

    Key properties:
    - All points on the same yard line → same X coordinate after transformation
    - All points on the same sideline → same Y coordinate after transformation
    - Preserves relative distances along each axis
    """

    def __init__(self, yard_line_angle: float, center_x: float = 960, center_y: float = 492):
        """
        Initialize the transformation.

        Args:
            yard_line_angle: Angle of yard lines in video (degrees from horizontal)
            center_x, center_y: Reference point for offset calculation
        """
        self.yard_line_angle = yard_line_angle
        self.center_x = center_x
        self.center_y = center_y

        # Field axes: Yard lines are horizontal in field coordinates
        # In video coordinates, they appear at yard_line_angle
        # Field X-axis (perpendicular to yard lines) is at yard_line_angle + 90°
        # Field Y-axis (parallel to yard lines) is at yard_line_angle

        field_x_angle_rad = np.radians(yard_line_angle + 90)
        field_y_angle_rad = np.radians(yard_line_angle)

        # Unit vectors for field axes (in video coordinates)
        self.field_x_u = np.array([np.cos(field_x_angle_rad), np.sin(field_x_angle_rad)])
        self.field_y_u = np.array([np.cos(field_y_angle_rad), np.sin(field_y_angle_rad)])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform a single point from video coordinates to field coordinates.

        The transformation projects the point onto the field axes:
        - new_x: Distance along field X-axis (perpendicular to yard lines)
        - new_y: Distance along field Y-axis (parallel to yard lines)

        Returns:
            (new_x, new_y) in field coordinates
        """
        # Vector from center to point
        dx = x - self.center_x
        dy = y - self.center_y

        # Project onto field axes
        field_x = dx * self.field_x_u[0] + dy * self.field_x_u[1]
        field_y = dx * self.field_y_u[0] + dy * self.field_y_u[1]

        return (float(field_x), float(field_y))

    def inverse_transform_point(self, field_x: float, field_y: float) -> Tuple[float, float]:
        """
        Transform from field coordinates back to video coordinates.

        Args:
            field_x, field_y: Coordinates in field space

        Returns:
            (x, y) in video coordinates
        """
        # Reconstruct vector from components
        dx = field_x * self.field_x_u[0] + field_y * self.field_y_u[0]
        dy = field_x * self.field_x_u[1] + field_y * self.field_y_u[1]

        x = dx + self.center_x
        y = dy + self.center_y

        return (float(x), float(y))

    def transform_trajectory(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform a list of points."""
        return [self.transform_point(x, y) for x, y in points]


def create_field_axis_transform(
    video_path: str,
    yard_line_angle: Optional[float] = None,
    detector: Optional[FieldOrientationDetector] = None
) -> FieldAxisProjectionTransform:
    """
    Create a field axis projection transform.

    This is the recommended way to create the transform for NFL All-22 footage.

    Args:
        video_path: Path to video file (for auto-detection)
        yard_line_angle: If known, provide directly (in degrees)
        detector: Existing detector instance (optional)

    Returns:
        FieldAxisProjectionTransform instance
    """
    from nfl_route_tracker.core.video_loader import VideoLoader

    if yard_line_angle is None:
        # Auto-detect from video
        with VideoLoader(video_path) as video:
            frame = video.get_frame(0)
            if frame is None:
                raise RuntimeError(f"Could not read first frame from {video_path}")

            if detector is None:
                detector = FieldOrientationDetector(
                    video_width=frame.shape[1],
                    video_height=frame.shape[0]
                )

            orientation = detector.detect_and_compute(frame)
            yard_line_angle = orientation.yard_line_angle

    return FieldAxisProjectionTransform(
        yard_line_angle=yard_line_angle,
        center_x=960,
        center_y=492
    )


class RotatedFieldAxisTransform:
    """
    Combined rotation + axis projection for better results.

    This combines:
    1. Initial rotation to roughly align with field
    2. Axis projection to ensure spatial relationships are preserved

    This handles cases where the simple axis projection might not be enough
    due to extreme camera angles.
    """

    def __init__(
        self,
        yard_line_angle: float,
        rotation_angle: float,
        center_x: float = 960,
        center_y: float = 492
    ):
        self.yard_line_angle = yard_line_angle
        self.rotation_angle = rotation_angle
        self.center_x = center_x
        self.center_y = center_y

        # Build rotation matrix
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        self.rotation_matrix = np.array([
            [cos_a, -sin_a, self.center_x * (1 - cos_a) + self.center_y * sin_a],
            [sin_a, cos_a, self.center_y * (1 - cos_a) - self.center_x * sin_a],
            [0, 0, 1]
        ])

        # Build axis projection matrix
        # Project onto axis perpendicular to yard lines (field X)
        field_x_angle = np.radians(yard_line_angle + 90)
        field_y_angle = np.radians(yard_line_angle)

        self.field_x_vec = np.array([np.cos(field_x_angle), np.sin(field_x_angle)])
        self.field_y_vec = np.array([np.cos(field_y_angle), np.sin(field_y_angle)])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a point: first rotate, then project."""
        # Step 1: Apply rotation
        rotated = self.rotation_matrix @ np.array([x, y, 1])
        rx, ry = rotated[0] / rotated[2], rotated[1] / rotated[2]

        # Step 2: Project onto field axes (relative to rotated center)
        dx = rx - self.center_x
        dy = ry - self.center_y

        field_x = dx * self.field_x_vec[0] + dy * self.field_x_vec[1]
        field_y = dx * self.field_y_vec[0] + dy * self.field_y_vec[1]

        return (float(field_x + self.center_x), float(field_y + self.center_y))


def test_transformation():
    """
    Test that the transformation preserves spatial relationships.
    """
    import matplotlib.pyplot as plt

    print("="*70)
    print("TESTING FIELD AXIS PROJECTION TRANSFORM")
    print("="*70)

    # Simulate the user's scenario:
    # Two players 'a' on same yardline, one player 'b' on different yardline

    transform = FieldAxisProjectionTransform(yard_line_angle=-2.9)

    # Test points representing players on same yardline
    same_yardline = [
        (1400, 300),  # Top-right player
        (1380, 305),  # Another on same yardline
        (1420, 298),  # Third on same yardline
    ]

    # Test points on different yardline
    diff_yardline = [
        (1400, 600),  # Lower player
    ]

    print("\nPoints on SAME yardline (should have similar X after transform):")
    for x, y in same_yardline:
        tx, ty = transform.transform_point(x, y)
        print(f"  ({x}, {y}) → ({tx:.1f}, {ty:.1f})")

    x_vals = [transform.transform_point(x, y)[0] for x, y in same_yardline]
    print(f"  X spread: {max(x_vals) - min(x_vals):.2f} pixels")

    print("\nPoint on DIFFERENT yardline (should have different X):")
    for x, y in diff_yardline:
        tx, ty = transform.transform_point(x, y)
        print(f"  ({x}, {y}) → ({tx:.1f}, {ty:.1f})")

    # Check if the transformation works
    same_yardline_x = [transform.transform_point(x, y)[0] for x, y in same_yardline]
    diff_yardline_x = transform.transform_point(*diff_yardline[0])[0]

    if max(same_yardline_x) - min(same_yardline_x) < 5:
        print("\n✅ SUCCESS: Points on same yardline have nearly identical X!")
    else:
        print("\n❌ FAILURE: Points on same yardline have different X!")

    return transform


if __name__ == "__main__":
    test_transformation()