#!/usr/bin/env python3
"""
NFL Route Tracker - Corrected Field Axis Projection
===================================================

FIXED VERSION: Accounts for vertically-slanted yard lines in All-22 footage

THE ISSUE:
- In All-22 footage from the sideline, yard lines appear VERTICALLY slanted
  (diagonal from top-left to bottom-right in the video)
- The old code assumed yard lines were nearly horizontal
- This caused the transformation to be applied incorrectly

THE FIX:
- Detect the TRUE orientation of yard lines (vertical in video = diagonal)
- Apply transformation that makes them VERTICAL (in field coordinates, horizontal)
- Ensure X-axis of output represents field DEPTH (yard line position)
- Ensure Y-axis of output represents field WIDTH (sideline position)

COORDINATE SYSTEM:
- Video coordinates: X = horizontal (left-right), Y = vertical (top-bottom)
- Field coordinates: X = depth (toward end zones), Y = width (sideline to sideline)

In All-22 footage:
- Moving toward end zones (vertical on field) = horizontal movement in video
- Moving sideline to sideline (horizontal on field) = vertical movement in video
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CorrectedFieldAxisTransform:
    """
    Transforms video coordinates to field coordinates for All-22 footage.

    KEY DIFFERENCE FROM OLD VERSION:
    - Old: Assumed yard lines are nearly horizontal in video
    - New: Detects that yard lines are VERTICALLY slanted (diagonal)
           and transforms them to be VERTICAL (field X = depth)
    """

    def __init__(
        self,
        yard_line_angle: float,  # Angle of yard lines in video (degrees)
        video_width: float = 1920,
        video_height: float = 984,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None
    ):
        """
        Initialize the transformation.

        Args:
            yard_line_angle: Angle of yard lines in video (from horizontal, degrees)
                             Positive = slants top-right to bottom-left
                             Negative = slants top-left to bottom-right
            video_width, video_height: Video dimensions for center calculation
            center_x, center_y: Center point for transformation (default: video center)
        """
        self.yard_line_angle = yard_line_angle
        self.video_width = video_width
        self.video_height = video_height
        self.center_x = center_x if center_x is not None else video_width / 2
        self.center_y = center_y if center_y is not None else video_height / 2

        # Convert angle to radians
        angle_rad = np.radians(yard_line_angle)

        # For All-22 footage with vertically-slanted yard lines:
        # - Field X (depth) corresponds to direction ALONG the yard lines (vertical in video)
        # - Field Y (width) corresponds to direction ACROSS the yard lines (horizontal in video)

        # Unit vector along yard lines (direction of field X = depth)
        # If yard lines slant at `yard_line_angle`, the along-yard direction is perpendicular
        along_yard_angle = angle_rad + np.pi / 2  # Perpendicular to yard line angle
        self.field_x_u = np.array([np.cos(along_yard_angle), np.sin(along_yard_angle)])

        # Unit vector across yard lines (direction of field Y = width)
        # This is parallel to the yard lines
        across_yard_angle = angle_rad
        self.field_y_u = np.array([np.cos(across_yard_angle), np.sin(across_yard_angle)])

        print(f"Transform initialized:")
        print(f"  Yard line angle: {yard_line_angle:.2f}°")
        print(f"  Field X (depth) direction: {along_yard_angle * 180 / np.pi:.1f}°")
        print(f"  Field Y (width) direction: {across_yard_angle * 180 / np.pi:.1f}°")

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform video coordinates to field coordinates.

        Args:
            x, y: Video coordinates (X = horizontal, Y = vertical)

        Returns:
            (field_x, field_y) where:
            - field_x = depth along field (toward end zones)
            - field_y = width across field (sideline to sideline)
        """
        # Vector from center to point
        dx = x - self.center_x
        dy = y - self.center_y

        # Project onto field axes
        # field_x: distance along yard lines (depth, toward end zones)
        # field_y: distance across yard lines (width, sideline to sideline)
        field_x = dx * self.field_x_u[0] + dy * self.field_x_u[1]
        field_y = dx * self.field_y_u[0] + dy * self.field_y_u[1]

        # Shift coordinates so center is at origin, then add offset for visualization
        # This makes the output more interpretable
        field_x_out = field_x + self.center_x
        field_y_out = field_y + self.center_y

        return (float(field_x_out), float(field_y_out))

    def transform_point_simple(self, x: float, y: float) -> Tuple[float, float]:
        """
        Simple transform - just projects onto field axes without centering.

        Returns:
            Raw projected coordinates (not shifted to video center)
        """
        dx = x - self.center_x
        dy = y - self.center_y

        field_x = dx * self.field_x_u[0] + dy * self.field_x_u[1]
        field_y = dx * self.field_y_u[0] + dy * self.field_y_u[1]

        return (float(field_x), float(field_y))

    def transform_trajectory(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform a list of points."""
        return [self.transform_point(x, y) for x, y in points]

    def transform_trajectory_simple(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform a list of points with simple output."""
        return [self.transform_point_simple(x, y) for x, y in points]


def create_corrected_transform(
    yard_line_angle: float,
    video_width: float = 1920,
    video_height: float = 984
) -> CorrectedFieldAxisTransform:
    """
    Create a corrected field axis transform for All-22 footage.

    Args:
        yard_line_angle: Angle of yard lines in video (degrees)
        video_width, video_height: Video dimensions

    Returns:
        CorrectedFieldAxisTransform instance
    """
    return CorrectedFieldAxisTransform(
        yard_line_angle=yard_line_angle,
        video_width=video_width,
        video_height=video_height
    )


def test_corrected_transform():
    """
    Test the corrected transformation with proper understanding
    of All-22 camera perspective.
    """
    print("="*70)
    print("TESTING CORRECTED FIELD AXIS TRANSFORM")
    print("="*70)
    print()
    print("Understanding the All-22 camera perspective:")
    print("  - Camera is on the SIDELINE, looking down the field")
    print("  - Yard lines appear as VERTICAL slanted lines in video")
    print("  - Player vertical movement in video = horizontal movement on field")
    print("  - Player horizontal movement in video = vertical movement on field")
    print()

    # Simulate yard line angle of -30 degrees (slants top-left to bottom-right)
    # This matches what we see in All-22 footage
    yard_line_angle = -30.0

    transform = CorrectedFieldAxisTransform(
        yard_line_angle=yard_line_angle,
        video_width=1920,
        video_height=984
    )

    # Test: Two players on the SAME yardline (same depth on field)
    # In video coordinates, they should have different Y positions (one higher, one lower)
    # but similar X positions
    same_yardline_players = [
        (1000, 200),  # Top player (on same yardline as below)
        (1000, 400),  # Bottom player (on same yardline as above)
    ]

    # Player on DIFFERENT yardline (different depth)
    diff_yardline_player = (1200, 300)

    print("TEST: Players on same yardline should have similar FIELD_X after transform")
    print()

    print("Input (Video Coordinates):")
    print(f"  Player A: ({same_yardline_players[0][0]}, {same_yardline_players[0][1]})")
    print(f"  Player B: ({same_yardline_players[1][0]}, {same_yardline_players[1][1]})")
    print(f"  Player C: ({diff_yardline_player[0]}, {diff_yardline_player[1]})")
    print()

    print("After CORRECTED transformation:")
    for i, (x, y) in enumerate(same_yardline_players):
        fx, fy = transform.transform_point(x, y)
        print(f"  Player {'AB'[i]}: field_x={fx:.1f}, field_y={fy:.1f}")

    fx_c, fy_c = transform.transform_point(*diff_yardline_player)
    print(f"  Player C: field_x={fx_c:.1f}, field_y={fy_c:.1f}")
    print()

    # Check: Players A and B (same yardline) should have similar field_x
    fx_a, _ = transform.transform_point(*same_yardline_players[0])
    fx_b, _ = transform.transform_point(*same_yardline_players[1])

    x_spread_same_yardline = abs(fx_a - fx_b)

    if x_spread_same_yardline < 20:
        print(f"✅ SUCCESS: Players on same yardline have similar field_x (diff={x_spread_same_yardline:.1f})")
    else:
        print(f"❌ FAILURE: Players on same yardline have different field_x (diff={x_spread_same_yardline:.1f})")

    return transform


if __name__ == "__main__":
    test_corrected_transform()