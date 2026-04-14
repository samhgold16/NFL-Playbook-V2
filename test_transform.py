#!/usr/bin/env python3
"""
NFL Route Tracker - Simple Transform Test
=========================================

This script tests the field transformation WITHOUT running the full pipeline.
It only requires the video file and outputs the detected angle and a simple test.

Usage:
    python test_transform_simple.py --video path/to/video.mp4
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nfl_route_tracker.tracking.fixed_field_orientation_detector import FixedFieldOrientationDetector
from nfl_route_tracker.tracking.final_field_transform import FinalFieldTransform


def test_transform(video_path: str):
    """Test the transform on a video."""
    print("="*60)
    print("NFL ROUTE TRACKER - SIMPLE TRANSFORM TEST")
    print("="*60)

    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read video: {video_path}")
        return False

    h, w = frame.shape[:2]
    print(f"\nVideo loaded: {w}x{h}")

    # Step 1: Detect yard line angle
    print("\n" + "-"*40)
    print("STEP 1: Detecting Yard Line Angle")
    print("-"*40)

    detector = FixedFieldOrientationDetector(
        video_width=w,
        video_height=h,
        canny_low=50,
        canny_high=150,
        hough_threshold=30,
        hough_min_line_length=60,
        hough_max_line_gap=20,
        angle_tolerance=45.0,
        min_field_lines=2
    )

    orientation = detector.detect_and_compute(frame)

    print(f"\n  Detected angle: {orientation.yard_line_angle:.1f}°")
    print(f"  Lines found: {orientation.all_lines_found}")
    print(f"  Confidence: {orientation.confidence:.2f}")

    # Check if angle is reasonable
    if 70 <= orientation.yard_line_angle <= 90:
        print("  ✅ Angle looks correct (~80°)")
    elif orientation.yard_line_angle < 10:
        print("  ❌ Angle is WRONG - detected nearly horizontal!")
        print("     Expected: ~80° for All-22 footage")
        return False
    else:
        print(f"  ⚠️ Angle is unusual: {orientation.yard_line_angle:.1f}°")

    # Step 2: Create and test transform
    print("\n" + "-"*40)
    print("STEP 2: Testing Transform")
    print("-"*40)

    transform = FinalFieldTransform(
        yard_line_angle=orientation.yard_line_angle,
        video_width=w,
        video_height=h
    )

    # Test on synthetic points that SHOULD be on same yardline
    # These points are COLLINEAR at the detected angle
    print("\n  Test: Points on SAME yardline (collinear at detected angle)")
    print("  (These points should have SIMILAR field_y after transform)")

    angle_rad = np.radians(orientation.yard_line_angle)
    slope = np.tan(angle_rad)

    # Create collinear points at detected angle
    collinear_points = []
    for x_frac in [0.3, 0.4, 0.5]:
        x = w * x_frac
        y = h * 0.7 + slope * (x - w * 0.5)  # On line at detected angle
        collinear_points.append((x, y))

    print(f"\n  Input points (collinear at {orientation.yard_line_angle:.1f}°):")
    transformed_ys = []
    for i, (x, y) in enumerate(collinear_points):
        fx, fy = transform.transform_point(x, y)
        print(f"    Point {i+1}: ({x:.0f}, {y:.0f}) -> field_y={fy:.1f}")
        transformed_ys.append(fy)

    y_spread = max(transformed_ys) - min(transformed_ys)
    print(f"\n  Y spread: {y_spread:.1f}px")

    if y_spread < 100:
        print("  ✅ PASS: Collinear points have similar field_y!")
        test1_pass = True
    else:
        print(f"  ❌ FAIL: Y spread too large ({y_spread:.1f}px)")
        test1_pass = False

    # Test 2: Points at different vertical positions
    print("\n  Test: Points at DIFFERENT depths")
    print("  (These should have DIFFERENT field_y values)")

    diff_points = [
        (w * 0.5, h * 0.3),
        (w * 0.5, h * 0.5),
        (w * 0.5, h * 0.7),
    ]

    print(f"\n  Input points (different Y = different depths):")
    transformed_ys_diff = []
    for i, (x, y) in enumerate(diff_points):
        fx, fy = transform.transform_point(x, y)
        print(f"    Point {i+1}: ({x:.0f}, {y:.0f}) -> field_y={fy:.1f}")
        transformed_ys_diff.append(fy)

    y_spread_diff = max(transformed_ys_diff) - min(transformed_ys_diff)
    print(f"\n  Y spread: {y_spread_diff:.1f}px")

    if y_spread_diff > 50:
        print("  ✅ PASS: Different depths have different field_y!")
        test2_pass = True
    else:
        print(f"  ⚠️ NOTE: Y spread is small ({y_spread_diff:.1f}px)")
        test2_pass = y_spread_diff > 30

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if test1_pass and test2_pass:
        print("✅ TRANSFORM IS WORKING CORRECTLY!")
        print()
        print("Next steps:")
        print("  1. Run the full pipeline:")
        print(f"     python detection_tracker_v2.py --video {video_path}")
        print()
        print("  2. Check the trajectory plot:")
        print("     View test_v3_output/*_trajectories.png")
        print()
        print("  3. Verify in the plot:")
        print("     - Players on SAME yardline should have SIMILAR Y")
        print("     - Offense/defense should show clear separation")
        return True
    else:
        print("❌ TRANSFORM NEEDS ADJUSTMENT")
        print()
        print("Test 1 (collinear = same yardline):", "PASS" if test1_pass else "FAIL")
        print("Test 2 (different depths):", "PASS" if test2_pass else "FAIL")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple field transform test"
    )
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to video file')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    success = test_transform(str(video_path))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
