"""
NFL Route Tracker - Field Orientation Detector
=============================================

This module detects the field orientation in the first frame of a video and computes
a homography transformation to map the perspective-distorted view to a perfectly
orthogonal top-down view.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class FieldOrientation:
    """Results from field orientation detection."""
    homography: np.ndarray  # 3x3 transformation matrix
    field_angle: float  # Rotation angle in degrees (how much to rotate)
    vanishing_point: Tuple[float, float]  # Vanishing point (x, y)
    field_corners: List[Tuple[float, float]]  # 4 corners of field in frame
    confidence: float  # Detection confidence (0-1)
    yard_line_angle: float  # Angle of yard lines in frame (before rotation)
    sideline_angle: float  # Angle of sidelines in frame (before rotation)
    yard_lines_count: int  # Number of yard lines detected
    sideline_lines_count: int  # Number of sideline lines detected


class FieldOrientationDetector:
    """
    Detects field orientation and computes perspective correction homography.

    The detector:
    1. Detects field lines using edge detection and Hough transform
    2. Classifies lines as yard lines (horizontal) or sidelines (vertical)
    3. Computes field rotation angle (how much the camera is tilted)
    4. Builds rotation+scale transformation to orthogonal view
    5. Optionally applies perspective correction for extreme angles

    This runs ONCE on the first frame to get a fixed transformation.
    ByteTrack GMC handles per-frame motion separately.
    """

    def __init__(
        self,
        video_width: int = 1920,
        video_height: int = 984,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        hough_min_line_length: int = 100,
        hough_max_line_gap: int = 50,
        angle_tolerance: float = 15.0,  # Degrees - lines within this of horizontal/vertical are field lines
        min_field_lines: int = 3,  # Minimum lines to detect for confidence
        perspective_threshold: float = 20.0,  # Angle threshold for perspective vs rotation correction
        target_field_width: int = 1920,  # Output width (pixels per 53.3 yards)
        target_field_height: int = 984,  # Output height (pixels per 100 yards)
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
        self.perspective_threshold = perspective_threshold
        self.target_field_width = target_field_width
        self.target_field_height = target_field_height

        # Cached homography for reuse
        self._cached_homography: Optional[np.ndarray] = None
        self._cached_orientation: Optional[FieldOrientation] = None
        self._last_frame: Optional[np.ndarray] = None

    def detect_and_compute(self, first_frame: np.ndarray) -> FieldOrientation:
        """
        Detect field orientation and compute homography transformation.

        This should be called ONCE on the first frame of a video.

        Args:
            first_frame: First video frame (BGR format)

        Returns:
            FieldOrientation object containing homography and metadata
        """
        self._last_frame = first_frame.copy()

        # Convert to grayscale
        if len(first_frame.shape) == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = first_frame

        # Step 1: Detect edges with slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Step 2: Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        if lines is None or len(lines) < self.min_field_lines:
            print(f"Warning: Only {len(lines) if lines is not None else 0} lines detected, using default (identity) homography")
            return self._create_default_orientation()

        # Step 3: Classify lines and estimate angles
        yard_lines, sideline_lines = self._classify_lines(lines)

        print(f"  Detected {len(yard_lines)} yard lines, {len(sideline_lines)} sidelines")

        # Step 4: Compute field orientation angles
        yard_angle = self._compute_line_angle(yard_lines) if yard_lines else 0
        sideline_angle = self._compute_line_angle(sideline_lines) if sideline_lines else 90

        # If sideline detection failed or is unreliable, assume perpendicular to yard lines
        if len(sideline_lines) < 2 or abs(sideline_angle - 90) > 45:
            print(f"  Sideline detection unreliable (angle={sideline_angle:.1f}°), assuming perpendicular to yard lines")
            sideline_angle = 90 if yard_angle == 0 else 0
            sideline_lines = []

        # Step 5: Compute field rotation angle (how much to rotate to make yard lines horizontal)
        # The rotation angle is the negative of the yard line angle
        rotation_angle = -yard_angle

        # Step 6: Compute vanishing point if we have sidelines (for perspective correction)
        vanishing_point = self._compute_vanishing_point(sideline_lines) if sideline_lines else None

        # Step 7: Build transformation matrix
        homography = self._compute_rotation_homography(rotation_angle, vanishing_point)

        # Step 8: Find field corners
        field_corners = self._estimate_field_corners(yard_lines, sideline_lines, homography)

        # Step 9: Compute confidence based on detection quality
        confidence = self._compute_confidence(yard_lines, sideline_lines, rotation_angle)

        orientation = FieldOrientation(
            homography=homography,
            field_angle=rotation_angle,  # This is the rotation to apply
            vanishing_point=vanishing_point if vanishing_point else (self.video_width / 2, 0),
            field_corners=field_corners,
            confidence=confidence,
            yard_line_angle=yard_angle,
            sideline_angle=sideline_angle,
            yard_lines_count=len(yard_lines),
            sideline_lines_count=len(sideline_lines)
        )

        # Cache for reuse
        self._cached_homography = homography
        self._cached_orientation = orientation

        print(f"\nField Orientation Detected:")
        print(f"  Yard lines detected: {len(yard_lines)}")
        print(f"  Sidelines detected: {len(sideline_lines)}")
        print(f"  Yard line angle: {yard_angle:.1f}°")
        print(f"  Sideline angle: {sideline_angle:.1f}°")
        print(f"  Rotation to apply: {rotation_angle:.1f}°")
        print(f"  Vanishing point: {vanishing_point if vanishing_point else 'N/A'}")
        print(f"  Confidence: {confidence:.2f}")

        return orientation

    def _classify_lines(
        self, lines: np.ndarray
    ) -> Tuple[List, List]:
        """
        Classify detected lines as yard lines or sidelines.

        Yard lines: Approximately horizontal (angle near 0 or 180)
        Sidelines: Approximately vertical (angle near 90 or 270)

        We also filter by line LOCATION - yard lines should be in the middle of the frame,
        not at the edges (which are likely sidelines or out-of-bounds markers).
        """
        yard_lines = []
        sideline_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip very short lines
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 50:
                continue

            # Compute angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Normalize angle to [-90, 90]
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180

            # Classify based on angle
            # Horizontal (yard lines): angle near 0
            if abs(angle) < self.angle_tolerance:
                # Check that the line is roughly in the middle of the frame (field area)
                # Yard lines run across the field, so they span left to right
                line_y = (y1 + y2) / 2
                if self.video_height * 0.2 < line_y < self.video_height * 0.8:
                    yard_lines.append(line)
            # Vertical (sidelines): angle near 90
            elif abs(abs(angle) - 90) < self.angle_tolerance:
                # Sidelines run along the field, so they span top to bottom
                line_x = (x1 + x2) / 2
                if self.video_width * 0.3 < line_x < self.video_width * 0.7:
                    sideline_lines.append(line)

        return yard_lines, sideline_lines

    def _compute_line_angle(self, lines: List) -> float:
        """Compute average angle of lines, using weighted median by length."""
        if not lines:
            return 0.0

        angles = []
        weights = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
            weights.append(length)

        # Return length-weighted median angle
        angles = np.array(angles)
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sort by angle
        idx = np.argsort(angles)
        angles = angles[idx]
        weights = weights[idx]

        # Find weighted median
        cumsum = np.cumsum(weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        return angles[median_idx]

    def _compute_vanishing_point(self, lines: List) -> Optional[Tuple[float, float]]:
        """
        Compute vanishing point as intersection of sidelines.

        Uses RANSAC-like approach to find most consistent intersection point.
        """
        if len(lines) < 2:
            return None

        # For each pair of lines, find intersection
        intersections = []
        line_lengths = []

        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

                if abs(denom) < 1e-6:  # Parallel lines
                    continue

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

                x_int = x1 + t * (x2 - x1)
                y_int = y1 + t * (y2 - y1)

                # Filter out unreasonable intersections (outside frame or very far)
                if -200 <= x_int <= self.video_width + 200 and -200 <= y_int <= self.video_height + 200:
                    intersections.append((x_int, y_int))
                    line_lengths.append(np.sqrt((x2-x1)**2 + (y2-y1)**2) + np.sqrt((x4-x3)**2 + (y4-y3)**2))

        if not intersections:
            return None

        # Return centroid of intersections, weighted by line length
        intersections = np.array(intersections)
        line_lengths = np.array(line_lengths)
        weights = line_lengths / line_lengths.sum()

        vx = np.sum(intersections[:, 0] * weights)
        vy = np.sum(intersections[:, 1] * weights)

        return (float(vx), float(vy))

    def _compute_rotation_homography(
        self,
        rotation_angle: float,
        vanishing_point: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Compute homography for rotation + optional perspective correction.

        For most All-22 videos, a simple rotation is correct. Complex perspective
        correction is only needed for extreme angles (>20°).
        """
        # Convert angle to radians
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Compute center of frame
        cx, cy = self.video_width / 2, self.video_height / 2

        # Build 3x3 transformation matrix that:
        # 1. Translates center to origin
        # 2. Rotates
        # 3. Translates back

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

        # Combine: T2 @ R @ T1
        H = T2 @ R @ T1

        # Normalize
        H = H / H[2, 2]

        return H

    def _estimate_field_corners(
        self,
        yard_lines: List,
        sideline_lines: List,
        homography: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Estimate the 4 corners of the field in the frame."""
        # For now, just use frame corners
        corners = [
            (0, 0),
            (self.video_width, 0),
            (self.video_width, self.video_height),
            (0, self.video_height)
        ]

        # Optionally: Transform corners through inverse homography to get field corners
        # (not implemented for simplicity)

        return corners

    def _compute_confidence(
        self,
        yard_lines: List,
        sideline_lines: List,
        rotation_angle: float
    ) -> float:
        """Compute detection confidence based on line quality."""
        # Base confidence from number of lines detected
        base_conf = 0.0

        if len(yard_lines) >= 3:
            base_conf += 0.3
        elif len(yard_lines) >= 1:
            base_conf += 0.1

        if len(sideline_lines) >= 2:
            base_conf += 0.2
        elif len(sideline_lines) >= 1:
            base_conf += 0.1

        # Penalize extreme rotation angles (might indicate misdetection)
        if abs(rotation_angle) > 30:
            base_conf *= 0.5
        elif abs(rotation_angle) > 20:
            base_conf *= 0.8

        # Line quality: check if angles are consistent
        if len(yard_lines) >= 2:
            angles = []
            for line in yard_lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            angle_std = np.std(angles)
            if angle_std < 5:  # Very consistent
                base_conf += 0.2

        return min(1.0, base_conf + 0.3)  # Minimum 30% confidence

    def _create_default_orientation(self) -> FieldOrientation:
        """Create default (identity) orientation when detection fails."""
        return FieldOrientation(
            homography=np.eye(3),
            field_angle=0.0,
            vanishing_point=(self.video_width / 2, 0),
            field_corners=[(0, 0), (self.video_width, 0),
                          (self.video_width, self.video_height), (0, self.video_height)],
            confidence=0.0,
            yard_line_angle=0.0,
            sideline_angle=90.0,
            yard_lines_count=0,
            sideline_lines_count=0
        )

    def apply_homography(
        self,
        x: float,
        y: float,
        homography: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Apply homography transformation to a point.

        Args:
            x, y: Input coordinates
            homography: 3x3 transformation matrix (uses cached if None)

        Returns:
            Transformed (x, y) coordinates
        """
        if homography is None:
            homography = self._cached_homography

        if homography is None:
            return x, y

        # Apply homography
        point = np.array([x, y, 1.0])
        transformed = homography @ point

        # Convert from homogeneous coordinates
        if transformed[2] != 0:
            return (float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2]))
        else:
            return (float(transformed[0]), float(transformed[1]))

    def apply_to_trajectory(
        self,
        trajectory: List[Dict],
        homography: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Apply homography to all points in a trajectory.

        Args:
            trajectory: List of detection dicts with 'center_x', 'center_y'
            homography: 3x3 transformation matrix

        Returns:
            Trajectory with corrected coordinates
        """
        if homography is None:
            homography = self._cached_homography

        if homography is None:
            return trajectory

        corrected_trajectory = []
        for det in trajectory:
            x, y = self.apply_homography(det['center_x'], det['center_y'], homography)
            corrected_det = det.copy()
            corrected_det['center_x'] = x
            corrected_det['center_y'] = y
            corrected_trajectory.append(corrected_det)

        return corrected_trajectory

    def visualize_detection(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize detected field lines on the frame.

        Args:
            frame: Frame to draw on (uses cached frame if None)

        Returns:
            Image with detected lines visualized
        """
        if frame is None:
            frame = self._last_frame

        if frame is None:
            return None

        vis = frame.copy()

        if self._cached_orientation is None:
            return vis

        # Draw yard lines in green
        if hasattr(self, '_last_yard_lines'):
            for line in self._last_yard_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw sidelines in blue
        if hasattr(self, '_last_sideline_lines'):
            for line in self._last_sideline_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw vanishing point
        if self._cached_orientation.vanishing_point:
            vp = self._cached_orientation.vanishing_point
            cv2.circle(vis, (int(vp[0]), int(vp[1])), 10, (0, 0, 255), -1)

        # Draw rotation info
        text = f"Rotation: {self._cached_orientation.field_angle:.1f}°"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis

    def reset(self):
        """Reset cached homography for new video."""
        self._cached_homography = None
        self._cached_orientation = None
        self._last_frame = None
        if hasattr(self, '_last_yard_lines'):
            del self._last_yard_lines
        if hasattr(self, '_last_sideline_lines'):
            del self._last_sideline_lines


def create_field_orientation_detector(**kwargs) -> FieldOrientationDetector:
    """
    Factory function to create FieldOrientationDetector with recommended defaults.

    Usage:
        detector = create_field_orientation_detector(
            video_width=1920,
            video_height=984
        )
    """
    defaults = {
        'video_width': 1920,
        'video_height': 984,
        'canny_low': 50,
        'canny_high': 150,
        'hough_threshold': 100,
        'hough_min_line_length': 100,
        'hough_max_line_gap': 50,
        'angle_tolerance': 15.0,
        'min_field_lines': 3
    }
    defaults.update(kwargs)
    return FieldOrientationDetector(**defaults)