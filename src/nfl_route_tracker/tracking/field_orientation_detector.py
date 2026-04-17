"""
NFL Route Tracker - FIXED Field Orientation Detector
=================================================
1. Detects the ACTUAL slanted yard lines (diagonal, not horizontal)
2. Computes the correct angle for transformation
3. Properly corrects for camera perspective
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

        # Check if endzone might be causing issues
        has_endzone = self._detect_endzone_presence(first_frame)
        if has_endzone:
            print("  Note: Endzone detected in frame - may affect line detection")

        # Find the dominant angle cluster (yard lines)
        yard_line_angle = self._find_dominant_angle(all_angles)

        print(f"  Dominant yard line angle: {yard_line_angle:.1f}°")

        # Validate the detected angle
        is_valid, validation_reason = self._is_valid_yard_line_angle(yard_line_angle)

        if not is_valid:
            print(f" WARNING: Detected angle may be incorrect!")
            print(f"     {validation_reason}")

            # If angle is exactly 0° or completely outside expected range,
            # skip the transformation entirely to avoid making things worse
            if yard_line_angle == 0 or yard_line_angle >= 40:
                print(f" SKIPPING field rotation transformation!")
                return self._create_default_orientation()
        else:
            print(f"  ✓ {validation_reason}")

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


        total_lines = sum(bins.values())

        # If we have ANY yard line range evidence (even just 2-3 lines),
        # and dominant is 0°, we should still consider the yard line angle
        if dominant_angle == 0 and yard_line_range_count > 0:
            # Use the best yard line angle if we have any evidence
            if best_yard_line_count >= 2:  # Lower threshold to 2 lines
                if len(yard_line_range_angles) > 1:
                    weighted_sum = sum(bin_angle * count for bin_angle, count in yard_line_range_angles)
                    return weighted_sum / yard_line_range_count
                return best_yard_line_angle

        # If dominant is in yard line range, use it
        if 65 <= dominant_angle <= 90 and dominant_angle != 0:
            return dominant_angle

        # If dominant is 0° and no yard line evidence, return 0°
        return dominant_angle

    def _is_valid_yard_line_angle(self, angle: float) -> Tuple[bool, str]:
        """
        Check if the detected angle is reasonable for All-22 footage.

        Returns:
            (is_valid, reason)
        """
        if angle == 0:
            return (False, "Angle is exactly 0° - likely detection of horizontal lines, not yard lines")

        if not (0 < angle < 90):
            return (False, f"Angle {angle}° is outside valid range (0-90°)")

        if angle > 45:
            return (False, f"Angle {angle}° is too close to vertical")

        # Valid range is 45-85° (with preference for 65-85°)
        if 0 <= angle <= 25:
            return (True, f"Angle {angle}° is in ideal range for All-22 yard lines")
        elif 25 <= angle < 45:
            return (True, f"Angle {angle}° is acceptable (though 65-85° is more common)")
        else:
            return (False, f"Angle {angle}° is suspicious")

    def _detect_endzone_presence(self, frame: np.ndarray) -> bool:
        """
        Detect if an endzone is visible in the frame.
        """
        h, w = frame.shape[:2]

        # Check the top portion of frame for bright white regions
        top_region = frame[:int(h * 0.2), :, :]

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = top_region

        # Calculate percentage of bright pixels
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size

        # If more than 30% of top region is bright white, likely endzone
        if bright_ratio > 0.30:
            return True

        # Check for rectangular patterns (endzone lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                              minLineLength=100, maxLineGap=20)

        if lines is not None:
            # Count horizontal lines in top region
            horizontal_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 10:  # Nearly horizontal
                    horizontal_count += 1

            # If many horizontal lines in top, likely endzone
            if horizontal_count > 5:
                return True

        return False

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

    def validate_orientation(self, first_frame: np.ndarray) -> Dict[str, any]:
        """
        Validate that the detected field orientation is correct.
        """
        # Use _cached_orientation (which is set after detect_and_compute)
        if self._cached_orientation is None:
            return {
                'is_valid': False,
                'message': 'No orientation detected yet. Run detect_and_compute() first.',
                'y_spread_same_yardline': -1,
                'y_spread_diff_depths': -1,
                'avg_line_angle': -1,
                'angle_deviation': -1,
                'lines_analyzed': 0
            }

        # Get the homography that was computed
        homography = self._cached_homography
        if homography is None:
            return {
                'is_valid': False,
                'message': 'No homography available. Run detect_and_compute() first.',
                'y_spread_same_yardline': -1,
                'y_spread_diff_depths': -1,
                'avg_line_angle': -1,
                'angle_deviation': -1,
                'lines_analyzed': 0
            }

        # Convert to grayscale
        if len(first_frame.shape) == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = first_frame

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        if lines is None or len(lines) < 2:
            return {
                'is_valid': False,
                'message': 'Not enough lines detected for validation',
                'y_spread_same_yardline': -1,
                'y_spread_diff_depths': -1,
                'avg_line_angle': -1,
                'angle_deviation': -1,
                'lines_analyzed': 0
            }

        # Apply detected homography to all line endpoints
        transformed_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Compute line length to filter short lines
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 50:  # Skip very short lines
                continue

            # Apply transformation
            def apply_transform(x, y):
                point = np.array([x, y, 1.0])
                transformed = homography @ point
                if abs(transformed[2]) > 1e-6:
                    return (float(transformed[0] / transformed[2]),
                            float(transformed[1] / transformed[2]))
                return (float(transformed[0]), float(transformed[1]))

            t_x1, t_y1 = apply_transform(x1, y1)
            t_x2, t_y2 = apply_transform(x2, y2)
            transformed_lines.append(((t_x1, t_y1), (t_x2, t_y2)))

        # Check 1: After rotation, lines should be nearly vertical (around 90°)
        angles_after_rotation = []
        for (x1, y1), (x2, y2) in transformed_lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 50:  # Skip short lines
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize to 0-90
                while angle < 0:
                    angle += 180
                while angle >= 180:
                    angle -= 180
                if angle >= 90:
                    angle = 180 - angle
                angles_after_rotation.append(angle)

        # Lines should be nearly vertical (around 90°)
        avg_angle = np.mean(angles_after_rotation) if angles_after_rotation else 0
        angle_deviation = abs(90 - avg_angle)

        # Check 2: Create synthetic test points on "same yardline" and verify Y consistency
        w, h = self.video_width, self.video_height
        yard_line_angle = self._cached_orientation.yard_line_angle

        # Apply same transform function
        def apply_transform(x, y):
            point = np.array([x, y, 1.0])
            transformed = homography @ point
            if abs(transformed[2]) > 1e-6:
                return (float(transformed[0] / transformed[2]),
                        float(transformed[1] / transformed[2]))
            return (float(transformed[0]), float(transformed[1]))

        # Create points collinear at the detected yard line angle
        m = np.tan(np.radians(yard_line_angle))
        b = h * 0.6  # Position in lower portion of frame

        test_points_same_yardline = [
            (w * 0.3, m * w * 0.3 + b),
            (w * 0.5, m * w * 0.5 + b),
            (w * 0.7, m * w * 0.7 + b),
        ]

        transformed_ys = []
        for x, y in test_points_same_yardline:
            t_x, t_y = apply_transform(x, y)
            transformed_ys.append(t_y)

        y_spread_same = max(transformed_ys) - min(transformed_ys) if len(transformed_ys) >= 2 else float('inf')

        # Check 3: Points at different depths should have different Y values
        test_points_diff_depth = [
            (w * 0.5, h * 0.3),  # Top (far)
            (w * 0.5, h * 0.5),  # Middle
            (w * 0.5, h * 0.7),  # Bottom (near)
        ]

        transformed_ys_diff = []
        for x, y in test_points_diff_depth:
            t_x, t_y = apply_transform(x, y)
            transformed_ys_diff.append(t_y)

        y_spread_diff = max(transformed_ys_diff) - min(transformed_ys_diff) if len(transformed_ys_diff) >= 2 else 0

        # Validation criteria:
        # 1. After rotation, average line angle should be close to 90° (vertical)
        # 2. Y spread for "same yardline" points should be small (< 50px)
        # 3. Y spread for "different depths" should be large (> 100px)

        is_valid = (
            angle_deviation < 20 and  # Lines should be near-vertical
            y_spread_same < 50 and     # Same yardline → similar Y
            y_spread_diff > 100        # Different depths → different Y
        )

        message = f"Orientation {'VALID' if is_valid else 'INVALID'}. "
        message += f"Avg line angle: {avg_angle:.1f}° (deviation from vertical: {angle_deviation:.1f}°). "
        message += f"Y spread same-yardline: {y_spread_same:.1f}px. "
        message += f"Y spread diff-depths: {y_spread_diff:.1f}px."

        return {
            'is_valid': is_valid,
            'message': message,
            'avg_line_angle': avg_angle,
            'angle_deviation': angle_deviation,
            'y_spread_same_yardline': y_spread_same,
            'y_spread_diff_depths': y_spread_diff,
            'lines_analyzed': len(transformed_lines)
        }


def create_fixed_detector(**kwargs) -> FixedFieldOrientationDetector:
    """Factory function for fixed detector."""
    return FixedFieldOrientationDetector(**kwargs)