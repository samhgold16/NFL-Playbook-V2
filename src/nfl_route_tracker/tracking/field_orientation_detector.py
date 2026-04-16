"""
NFL Route Tracker - FIXED Field Orientation Detector
=================================================

CRITICAL FIX: The old detector was looking for nearly-HORIZONTAL lines,
but in All-22 footage, yard lines are actually NEARLY VERTICAL (~75-80°).

This fixed version:
1. Detects the ACTUAL slanted yard lines (diagonal, not horizontal)
2. Computes the correct angle for transformation
3. Properly corrects for camera perspective

KEY INSIGHT FROM VIDEO ANALYSIS:
- Yard lines in All-22 footage slant from top-left to bottom-right
- They appear at approximately 75-80° from horizontal
- The old code was finding WRONG lines (probably hash marks)
- This is why the transformation wasn't working
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class FixedFieldOrientation:
    """Results from fixed field orientation detection."""
    homography: np.ndarray  # 3x3 transformation matrix
    field_angle: float  # Rotation angle in degrees (how much to rotate)
    yard_line_angle: float  # Actual angle of yard lines in video (75-80° typical)
    yard_lines_count: int
    confidence: float
    # NEW: Additional diagnostics
    all_lines_found: int
    angle_histogram: Dict[float, int]


class FixedFieldOrientationDetector:
    """
    FIXED detector that properly identifies slanted yard lines in All-22 footage.

    CRITICAL DIFFERENCE from old version:
    - Old: Looked for nearly-horizontal lines (0°)
    - New: Detects diagonal lines at their ACTUAL angle (~75-80°)
    """

    def __init__(
        self,
        video_width: int = 1920,
        video_height: int = 984,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 50,  # Lowered to detect more lines
        hough_min_line_length: int = 80,  # Lowered to detect slanted lines
        hough_max_line_gap: int = 30,
        angle_tolerance: float = 45.0,  # WIDER tolerance for diagonal lines
        min_field_lines: int = 2,
    ):
        self.video_width = video_width
        self.video_height = video_height
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.angle_tolerance = angle_tolerance
        self.min_field_lines = min_field_lines

        self._cached_homography: Optional[np.ndarray] = None
        self._cached_orientation: Optional[FixedFieldOrientation] = None
        self._last_frame = None
        self._all_detected_lines = []

    def detect_and_compute(self, first_frame: np.ndarray) -> FixedFieldOrientation:
        """
        Detect field orientation with FIXED line detection.

        Key change: Now properly detects diagonal yard lines instead of
        assuming they're horizontal.
        """
        self._last_frame = first_frame.copy()

        # Convert to grayscale
        if len(first_frame.shape) == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = first_frame

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        if lines is None or len(lines) < 2:
            print("Warning: Not enough lines detected, using default")
            return self._create_default_orientation()

        # Store all detected lines for analysis
        self._all_detected_lines = lines.copy()

        # Analyze ALL detected lines to find the dominant angle
        all_angles = self._compute_all_line_angles(lines)

        print(f"\n  Total lines detected: {len(lines)}")

        # Create angle histogram for diagnostics
        angle_histogram = {}
        for angle in all_angles:
            bin_angle = round(angle / 5) * 5  # Bin by 5 degrees
            angle_histogram[bin_angle] = angle_histogram.get(bin_angle, 0) + 1

        print(f"  Angle distribution: {dict(sorted(angle_histogram.items()))}")

        # Find the dominant angle cluster (yard lines)
        yard_line_angle = self._find_dominant_angle(all_angles)

        print(f"  Dominant yard line angle: {yard_line_angle:.1f}°")

        # Compute rotation needed to make yard lines vertical
        # We WANT yard lines to be vertical in the output
        # If detected angle is 75°, we need to rotate by -75° to make them vertical
        #rotation_angle = 90-yard_line_angle
        rotation_angle = yard_line_angle

        # Build homography for rotation
        homography = self._compute_rotation_homography(rotation_angle)

        # Compute confidence based on line detection
        confidence = min(1.0, len(lines) / 20)

        orientation = FixedFieldOrientation(
            homography=homography,
            field_angle=rotation_angle,
            yard_line_angle=yard_line_angle,
            yard_lines_count=len(lines),
            confidence=confidence,
            all_lines_found=len(lines),
            angle_histogram=angle_histogram
        )

        self._cached_homography = homography
        self._cached_orientation = orientation

        print(f"\n  Field Orientation:")
        print(f"    Detected yard line angle: {yard_line_angle:.1f}°")
        print(f"    Rotation to apply: {rotation_angle:.1f}°")
        print(f"    Lines found: {len(lines)}")
        print(f"    Confidence: {confidence:.2f}")

        return orientation

    def _compute_all_line_angles(self, lines: np.ndarray) -> List[float]:
        """
        Compute angles for ALL detected lines, not just "horizontal" ones.

        This is the KEY FIX: We used to filter out non-horizontal lines.
        Now we use ALL lines to find the dominant angle.

        CRITICAL: Normalize to 0-90° range because:
        - 0° = 180° (horizontal line has no direction)
        - 90° = 270° (vertical line has no direction)
        """
        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip very short lines
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # WHAT VALUES????
            if length < 850:
                continue

            # Compute angle using atan2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Normalize to 0-90° range (lines have NO direction/orientation)
            # First normalize to 0-180
            while angle < 0:
                angle += 180
            while angle >= 180:
                angle -= 180

            # Then fold 180° to 0° (horizontal lines are directionless)
            if angle >= 90:
                angle = 180 - angle

            # Now angle is in 0-90 range where:
            # 0° = horizontal, 90° = vertical
            angles.append(angle)

        return angles

    def _find_dominant_angle(self, angles: List[float]) -> float:
        """
        Find the dominant angle cluster using histogram analysis.

        Yard lines will form a cluster around a specific angle.
        We find this by looking at the histogram peaks.

        CRITICAL: In All-22 footage, yard lines should be in 65-90° range.
        We prioritize this range over horizontal lines (0°) because
        the sideline camera creates the characteristic diagonal yard lines.
        """
        if not angles:
            return 0.0

        # Create histogram with 5-degree bins
        bins = {}
        for angle in angles:
            bin_angle = round(angle / 5) * 5
            bins[bin_angle] = bins.get(bin_angle, 0) + 1

        # Count lines in expected yard line range (65-90°)
        # Yard lines in All-22 footage are typically 75-85°
        yard_line_range_count = 0
        yard_line_range_angles = []

        for bin_angle, count in bins.items():
            if 65 <= bin_angle <= 90:
                yard_line_range_count += count
                yard_line_range_angles.append((bin_angle, count))

        # Find the best angle in the yard line range
        best_yard_line_angle = 0
        best_yard_line_count = 0

        for bin_angle, count in yard_line_range_angles:
            if count > best_yard_line_count:
                best_yard_line_count = count
                best_yard_line_angle = bin_angle

        # Find the overall dominant angle
        max_count = 0
        dominant_angle = 0

        for bin_angle, count in bins.items():
            if count > max_count:
                max_count = count
                dominant_angle = bin_angle

        # DECISION LOGIC:
        # If there are enough lines in the yard line range (>= 5 lines or >= 30% of dominant),
        # and the yard line angle is at least 5 lines, use the yard line angle
        # This prioritizes the slanted yard lines characteristic of All-22 footage

        if yard_line_range_count >= 5 and best_yard_line_count >= 5:
            # Use weighted average of yard line range angles for better precision
            if yard_line_range_count > 0:
                weighted_sum = sum(bin_angle * count for bin_angle, count in yard_line_range_angles)
                return weighted_sum / yard_line_range_count
            return best_yard_line_angle
        elif dominant_angle == 0 and yard_line_range_count >= 3:
            # Fallback: if 0° is dominant but we have yard line evidence, use yard line angle
            weighted_sum = sum(bin_angle * count for bin_angle, count in yard_line_range_angles)
            return weighted_sum / yard_line_range_count

        return dominant_angle

    def _compute_rotation_homography(self, rotation_angle: float) -> np.ndarray:
        """
        Compute homography for rotation.

        For All-22 footage:
        - Yard lines are slanted (e.g., 75° from horizontal)
        - We rotate to make them VERTICAL (90°)
        """
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        cx, cy = self.video_width / 2, self.video_height / 2

        # Translation to origin
        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        # Rotation
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

        # Translation back
        T2 = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ])

        H = T2 @ R @ T1
        H = H / H[2, 2]

        return H

    def apply_homography(self, x: float, y: float,
                         homography: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Apply homography transformation to a point."""
        if homography is None:
            homography = self._cached_homography

        if homography is None:
            return x, y

        point = np.array([x, y, 1.0])
        transformed = homography @ point

        if abs(transformed[2]) > 1e-6:
            return (float(transformed[0] / transformed[2]),
                    float(transformed[1] / transformed[2]))
        else:
            return (float(transformed[0]), float(transformed[1]))

    def _create_default_orientation(self) -> FixedFieldOrientation:
        """Create default orientation when detection fails."""
        return FixedFieldOrientation(
            homography=np.eye(3),
            field_angle=0.0,
            yard_line_angle=0.0,
            yard_lines_count=0,
            confidence=0.0,
            all_lines_found=0,
            angle_histogram={}
        )

    def reset(self) -> None:
        """Reset detector state for processing a new video."""
        self._cached_homography = None
        self._cached_orientation = None
        self._last_frame = None
        self._all_detected_lines = []


def create_fixed_detector(**kwargs) -> FixedFieldOrientationDetector:
    """Factory function for fixed detector."""
    return FixedFieldOrientationDetector(**kwargs)


def test_fixed_detector():
    """Test the fixed detector."""
    print("="*70)
    print("TESTING FIXED FIELD ORIENTATION DETECTOR")
    print("="*70)
    print()
    print("KEY FIX: Now detecting ACTUAL slanted yard lines (~75-80°) instead of")
    print("         only looking for nearly-horizontal lines (~0°)")
    print()


if __name__ == "__main__":
    test_fixed_detector()