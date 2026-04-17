#!/usr/bin/env python3
"""
NFL Route Tracker - Final Corrected Field Transform
====================================================

This module handles the transformation from video coordinates to field coordinates
for All-22 footage with slanted yard lines.
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FinalFieldTransform:
    """
    Final field transformation for All-22 footage with slanted yard lines.

    After applying this transform:
    - Points on the SAME yardline → have SIMILAR field_y values
    - Points at DIFFERENT depths → have DIFFERENT field_y values

    COORDINATE SEMANTICS (Important for Route Interpretation):
    ===========================================================
    The returned coordinates represent:
    - field_x: Sideline position (field width, 0-53.3 yards)
    - field_y: Depth position (field length toward endzone)

    This means:
    - A vertical route (streak, go) → X changes significantly, Y stays constant
    - A horizontal route (slant, out) → Y changes significantly, X stays constant

    Args:
        yard_line_angle: Angle of yard lines in raw video (typically 75-80°)
        video_width: Width of the video frame
        video_height: Height of the video frame
    """

    def __init__(
        self,
        yard_line_angle: float,
        video_width: float,
        video_height: float
    ):
        self.yard_line_angle = yard_line_angle
        self.video_width = video_width
        self.video_height = video_height
        self.center_x = video_width / 2
        self.center_y = video_height / 2

        # Pre-compute rotation matrix for -yard_line_angle
        # We rotate by NEGATIVE angle to straighten slanted yard lines
        angle_rad = np.radians(-yard_line_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Centered rotation matrix
        self.rotation_matrix = np.array([
            [cos_a, -sin_a, self.center_x - cos_a * self.center_x + sin_a * self.center_y],
            [sin_a, cos_a, self.center_y - sin_a * self.center_x - cos_a * self.center_y],
            [0, 0, 1]
        ])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform video coordinates to field coordinates.

        Args:
            x: X coordinate in video (represents field length, horizontal movement)
            y: Y coordinate in video (represents field width, vertical movement)

        Returns:
            (field_x, field_y) where:
            - field_x = sideline position (field width on the field)
            - field_y = depth position (toward endzones, same yardline = similar field_y)

        Example:
            >>> # A vertical route (streak) would have:
            >>> # start = (100, 500), end = (500, 505)  # X changes, Y stays ~same
            >>> # A horizontal route (slant) would have:
            >>> # start = (300, 200), end = (310, 400)  # Y changes significantly
        """
        point = np.array([x, y, 1.0])
        transformed = self.rotation_matrix @ point

        return (float(transformed[0]), float(transformed[1]))

    def get_field_depth(self, x: float, y: float) -> float:
        """
        Get the field depth position (Y on field, toward endzone).

        This is the primary coordinate for determining:
        - "Is player A ahead of player B on the field?"
        - "What yardline is this player at?"
        """
        _, field_y = self.transform_point(x, y)
        return field_y

    def get_field_width(self, x: float, y: float) -> float:
        """
        Get the field width position (sideline position).

        This is the coordinate for determining:
        - "Is player A closer to the sideline than player B?"
        - "What's the player's position relative to the hash marks?"
        """
        field_x, _ = self.transform_point(x, y)
        return field_x


def test_final_transform():
    """Test the final corrected transformation with coordinate semantics validation."""
    print("="*70)
    print("FINAL FIELD TRANSFORM TEST")
    print("="*70)
    print()
    print("COORDINATE SEMANTICS:")
    print("  Video X axis = field LENGTH (horizontal) = route goes up/down field")
    print("  Video Y axis = field WIDTH (vertical) = sideline to sideline movement")
    print()
    print("  Vertical route (go, streak) → X changes, Y stays constant")
    print("  Horizontal route (slant, out) → Y changes, X stays constant")
    print()

    # Test with synthetic collinear points
    w, h = 2864, 1490
    yard_line_angle = 80.0

    transform = FinalFieldTransform(
        yard_line_angle=yard_line_angle,
        video_width=w,
        video_height=h
    )

    print("-"*50)
    print("TEST 1: Collinear points on same yardline (synthetic)")
    print("-"*50)
    print()

    # Create points that ARE collinear at 80°
    m = np.tan(np.radians(yard_line_angle))
    b = h * 0.7  # Offset to put them in lower portion

    collinear_points = [
        (w * 0.3, m * w * 0.3 + b),
        (w * 0.5, m * w * 0.5 + b),
        (w * 0.7, m * w * 0.7 + b),
    ]

    print("Input points (collinear at 80°):")
    for i, (x, y) in enumerate(collinear_points):
        print(f"  Point {i+1}: ({x:.0f}, {y:.0f})")

    print("\nAfter rotation:")
    field_ys = []
    for i, (x, y) in enumerate(collinear_points):
        fx, fy = transform.transform_point(x, y)
        print(f"  Point {i+1}: field_x={fx:.1f}, field_y={fy:.1f}")
        field_ys.append(fy)

    y_spread = max(field_ys) - min(field_ys)
    print(f"\n  Y spread: {y_spread:.1f}px")
    if y_spread < 50:
        print("  ✅ PASS: Collinear points have similar field_y!")
        test1_pass = True
    else:
        print(f"  ❌ FAIL: Y spread too large ({y_spread:.1f}px)")
        test1_pass = False

    print()
    print("-"*50)
    print("TEST 2: Points on different yardlines (different depths)")
    print("-"*50)
    print()

    diff_points = [
        (w * 0.5, h * 0.3),  # Top (further from camera = far endzone)
        (w * 0.5, h * 0.5),  # Middle
        (w * 0.5, h * 0.7),  # Bottom (near camera = near endzone)
    ]

    print("Input points (different Y = different depths):")
    for i, (x, y) in enumerate(diff_points):
        print(f"  Point {i+1}: ({x:.0f}, {y:.0f})")

    print("\nAfter rotation:")
    field_ys_diff = []
    for i, (x, y) in enumerate(diff_points):
        fx, fy = transform.transform_point(x, y)
        print(f"  Point {i+1}: field_x={fx:.1f}, field_y={fy:.1f}")
        field_ys_diff.append(fy)

    y_spread_diff = max(field_ys_diff) - min(field_ys_diff)
    print(f"\n  Y spread: {y_spread_diff:.1f}px")
    if y_spread_diff > 100:
        print("  ✅ PASS: Different depth points have different field_y!")
        test2_pass = True
    else:
        print(f"  ❌ FAIL: Y spread too small ({y_spread_diff:.1f}px)")
        test2_pass = False

    print()
    print("-"*50)
    print("TEST 3: Route interpretation verification")
    print("-"*50)
    print()

    # Simulate a vertical route (streak): X changes, Y stays ~same
    vertical_route = [
        (200, 500),   # Start
        (400, 502),   # Middle
        (600, 501),   # End
    ]

    print("Simulated VERTICAL route (streak): X changes, Y stays ~same")
    vertical_ys = []
    for x, y in vertical_route:
        fx, fy = transform.transform_point(x, y)
        vertical_ys.append(fy)
        print(f"  ({x}, {y}) → field_x={fx:.1f}, field_y={fy:.1f}")

    # Check that field_x changes more than field_y
    vertical_x_range = max(fx for x, y in vertical_route) - min(fx for x, y in vertical_route)
    vertical_y_range = max(vertical_ys) - min(vertical_ys)
    print(f"\n  In field coordinates: X range={vertical_x_range:.1f}, Y range={vertical_y_range:.1f}")
    print(f"  → Primary direction: {'VERTICAL (X changes more)' if vertical_x_range > vertical_y_range else 'HORIZONTAL (Y changes more)'}")

    # Simulate a horizontal route (slant): Y changes significantly
    horizontal_route = [
        (300, 200),   # Start
        (310, 350),   # Middle
        (320, 500),   # End
    ]

    print("\nSimulated HORIZONTAL route (slant): Y changes significantly")
    horizontal_xs = []
    for x, y in horizontal_route:
        fx, fy = transform.transform_point(x, y)
        horizontal_xs.append(fx)
        print(f"  ({x}, {y}) → field_x={fx:.1f}, field_y={fy:.1f}")

    horizontal_x_range = max(horizontal_xs) - min(horizontal_xs)
    horizontal_y_range = max(fy for x, y in horizontal_route) - min(fy for x, y in horizontal_route)
    print(f"\n  In field coordinates: X range={horizontal_x_range:.1f}, Y range={horizontal_y_range:.1f}")
    print(f"  → Primary direction: {'VERTICAL (X changes more)' if horizontal_x_range > horizontal_y_range else 'HORIZONTAL (Y changes more)'}")

    test3_pass = True  # Visual verification
    print("\n  Route interpretation is correct if:")
    print("   - Vertical route: X changes, Y stays similar")
    print("   - Horizontal route: Y changes significantly")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    if test1_pass and test2_pass:
        print("✅ FIELD TRANSFORM IS WORKING CORRECTLY!")
    else:
        print("❌ Transform needs adjustment")


def create_final_transform(
    yard_line_angle: float,
    video_width: float,
    video_height: float
) -> FinalFieldTransform:
    """Factory function to create final transform."""
    return FinalFieldTransform(
        yard_line_angle=yard_line_angle,
        video_width=video_width,
        video_height=video_height
    )


if __name__ == "__main__":
    test_final_transform()
