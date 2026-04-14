#!/usr/bin/env python3
"""
NFL Route Tracker - Final Corrected Field Transform
====================================================

THE CORRECT SOLUTION:

After extensive testing, the key insight is:

For All-22 footage with ~80° yard line slant:
1. Rotate by -80° to straighten the field
2. After rotation:
   - rotated_Y ≈ 0.985*old_X (dominated by original X!)
   - rotated_X ≈ 0.985*old_Y (dominated by original Y!)

The ROTATED_Y is what represents DEPTH on the field!
This is because 80° rotation swaps the axes:
- Original X → becomes nearly vertical (Y in rotated space)
- Original Y → becomes nearly horizontal (X in rotated space)

So after rotation:
- rotated_Y = field_depth (players on same yardline → similar rotated_Y)
- rotated_X = field_width (sideline position)

This is the CORRECT transform!
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FinalFieldTransform:
    """
    FINAL CORRECTED field transformation.

    For All-22 footage with ~80° yard line slant:
    1. Rotate by -yard_line_angle to straighten the field
    2. Use rotated_Y as field_depth (same yardline = similar rotated_Y)
    3. Use rotated_X as field_width (sideline position)
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

        Returns:
            (field_x, field_y) where:
            - field_x = rotated_X = sideline position (width on field)
            - field_y = rotated_Y = depth position (toward end zones)
                          ★ SAME yardline → SIMILAR field_y ★
        """
        point = np.array([x, y, 1.0])
        transformed = self.rotation_matrix @ point

        return (float(transformed[0]), float(transformed[1]))

    def get_field_depth(self, x: float, y: float) -> float:
        """Get the field depth (Y position on same yardline)."""
        _, field_y = self.transform_point(x, y)
        return field_y

    def get_field_width(self, x: float, y: float) -> float:
        """Get the field width (sideline position)."""
        field_x, _ = self.transform_point(x, y)
        return field_x


def test_final_transform():
    """Test the final corrected transformation."""
    print("="*70)
    print("FINAL CORRECTED FIELD TRANSFORM TEST")
    print("="*70)
    print()
    print("CORRECT APPROACH:")
    print("1. Rotate by -yard_line_angle (e.g., -80°)")
    print("2. rotated_Y = field_depth (same yardline → similar rotated_Y)")
    print("3. rotated_X = field_width (sideline position)")
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
    print("TEST 2: Points on different yardlines")
    print("-"*50)
    print()

    diff_points = [
        (w * 0.5, h * 0.3),  # Top (further)
        (w * 0.5, h * 0.5),  # Middle
        (w * 0.5, h * 0.7),  # Bottom (near camera)
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
    print("="*70)
    print("SUMMARY")
    print("="*70)
    if test1_pass and test2_pass:
        print("✅ FINAL TRANSFORM IS WORKING CORRECTLY!")
        print()
        print("The transform correctly:")
        print("  - Points on same yardline (collinear at yard_line_angle)")
        print("    → have SIMILAR rotated_Y (field_depth)")
        print("  - Points on different depths")
        print("    → have DIFFERENT rotated_Y (field_depth)")
        print()
        print("USE THIS TRANSFORM IN THE MAIN PIPELINE!")
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
