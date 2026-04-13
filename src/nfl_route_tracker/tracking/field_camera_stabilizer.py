"""
NFL Route Tracker - Field-Specific Camera Stabilizer
===================================================

This module improves camera stabilization by focusing on field-specific features instead of general scene features.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore


class FieldFeatureDetector:
    """
    Detects field-specific features in NFL All-22 footage.

    Focuses on:
    - White yard lines and hash marks
    - Field markings (sidelines, numbers)
    - High-contrast edges between white and green
    """

    def __init__(self):
        # Field color ranges in HSV (white/off-white detection)
        self.field_lower_white = np.array([0, 0, 200])
        self.field_upper_white = np.array([180, 30, 255])

        # Green field color range
        self.field_green_lower = np.array([35, 40, 40])
        self.field_green_upper = np.array([85, 255, 200])

    def detect_field_features(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect field-specific features in a frame.

        Returns:
            - Keypoints for tracking (as numpy array for optical flow)
            - Binary mask of field features
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Edge detection for line features
        edges = cv2.Canny(gray, 50, 150)

        # Method 2: Detect white markings
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.field_lower_white, self.field_upper_white)

        # Method 3: Detect horizontal lines (yard lines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)

        # Combine features
        field_features = cv2.bitwise_or(edges, white_mask)
        field_features = cv2.bitwise_or(field_features, horizontal_lines)

        # Find keypoints on field features
        keypoints = self._extract_keypoints_from_mask(gray, field_features)

        return keypoints, field_features

    def _extract_keypoints_from_mask(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from areas with field features.

        Uses goodFeaturesToTrack for corner-like features on field markings.
        Returns numpy array for compatibility with optical flow.
        """
        # Apply mask to gray image
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Find corners/features using Shi-Tomasi
        keypoints = cv2.goodFeaturesToTrack(
            masked_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
            mask=mask
        )

        if keypoints is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        return keypoints


class FieldCameraStabilizer:
    """
    Camera stabilizer using field-specific features.

    This is an ALTERNATIVE to CameraStabilizer when:
    - ByteTrack's built-in GMC is not being used
    - More accurate field-based stabilization is needed
    - Camera panning is causing trajectory drift

    The stabilizer:
    1. Detects field features in each frame
    2. Matches features between consecutive frames
    3. Estimates homography transformation
    4. Smooths the transformation to reduce jitter
    5. Applies transformation to stabilize coordinates

    Provides the same interface as CameraStabilizer for easy swapping.
    """

    def __init__(
        self,
        max_features: int = 200,
        smoothing_window: int = 10,
        ransac_threshold: float = 3.0,
        motion_threshold: float = 1.0,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.max_features = max_features
        self.smoothing_window = smoothing_window
        self.ransac_threshold = ransac_threshold
        self.motion_threshold = motion_threshold

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[np.ndarray] = None
        self._frame_count: int = 0

        # Feature detector (ORB for fallback)
        self._feature_detector = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Field-specific feature detector
        self._field_detector = FieldFeatureDetector()

        # Transformation smoothing and cumulative homography
        self._transform_history: List[Tuple[float, float, float]] = []
        self._cumulative_H: np.ndarray = np.eye(3, dtype=np.float64)
        self._cumulative_inv_H: np.ndarray = np.eye(3, dtype=np.float64)
        self._total_motion_pixels: float = 0.0

        # LK optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    # =========================================================================
    # Helper methods (same interface as CameraStabilizer)
    # =========================================================================

    def is_ready(self) -> bool:
        """Check if stabilizer has processed enough frames."""
        return self._frame_count > 1

    def get_motion_stats(self) -> dict:
        """Get camera motion statistics."""
        return {
            'total_motion_pixels': self._total_motion_pixels,
            'avg_motion_per_frame': self._total_motion_pixels / max(1, self._frame_count - 1),
            'frames_processed': self._frame_count
        }

    def get_transform(self) -> Optional[np.ndarray]:
        """Cumulative forward homography H_{0 -> t} (current frame in frame-0 space)."""
        return self._cumulative_H.copy() if self._frame_count > 1 else None

    def get_inverse_transform(self) -> Optional[np.ndarray]:
        """Cumulative inverse homography H_{t -> 0} used for stabilization."""
        return self._cumulative_inv_H.copy() if self._frame_count > 1 else None

    # =========================================================================
    # Coordinate stabilization methods
    # =========================================================================

    def stabilize_bbox(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """
        Apply the cumulative inverse homography to a bounding box.
        """
        if not self.is_ready():
            return x, y, w, h

        # Four corners as homogeneous column vectors (3 × 4)
        corners = np.array([
            [x,     y,     1.0],
            [x + w, y,     1.0],
            [x,     y + h, 1.0],
            [x + w, y + h, 1.0],
        ], dtype=np.float64).T

        transformed = self._cumulative_inv_H @ corners
        transformed /= transformed[2]

        min_x = float(np.min(transformed[0]))
        max_x = float(np.max(transformed[0]))
        min_y = float(np.min(transformed[1]))
        max_y = float(np.max(transformed[1]))

        return min_x, min_y, max_x - min_x, max_y - min_y

    def stabilize_trajectory_detection(self, detection: Detection) -> Detection:
        """
        Stabilize a trajectory Detection to first-frame coordinates.
        """
        sx, sy, sw, sh = self.stabilize_bbox(detection.x, detection.y, detection.width, detection.height)
        return Detection(
            frame_id=detection.frame_id,
            x=sx, y=sy, width=sw, height=sh,
            confidence=detection.confidence
        )

    def stabilize_trajectory_store(self, store: TrajectoryStore) -> TrajectoryStore:
        """
        Stabilize every detection in every trajectory to first-frame coordinates.
        """
        if not self.enabled:
            return store

        stabilized_store = TrajectoryStore()

        for trajectory in store.get_all_trajectories():
            for detection in trajectory.detections:
                stabilized_det = self.stabilize_trajectory_detection(detection)
                stabilized_store.add_detection(trajectory.track_id, stabilized_det)

        return stabilized_store

    # =========================================================================
    # Main update loop
    # =========================================================================

    def update(self, frame: np.ndarray) -> None:
        """
        Update stabilizer with new frame (call after processing each frame).
        """
        if not self.enabled:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # First frame: store reference, detect initial features
        if self._frame_count == 0:
            self._prev_gray = gray.copy()
            # Extract initial keypoints using field features
            keypoints = self._field_detector.detect_field_features(frame)[0]
            if len(keypoints) > 0:
                self._prev_keypoints = keypoints
            else:
                # Fallback to general feature detection
                self._prev_keypoints = self._extract_general_features(gray)
            self._frame_count += 1
            return

        # Subsequent frames: track → estimate → compose
        matched_prev, matched_curr = self._track_features(gray)

        if len(matched_prev) >= 4:
            H_frame = self._estimate_homography(matched_prev, matched_curr)

            if H_frame is not None:
                H_smoothed = self._smooth_transform(H_frame)

                # Compose into cumulative homography H_{0 -> t}
                self._cumulative_H = H_smoothed @ self._cumulative_H
                self._cumulative_inv_H = np.linalg.inv(self._cumulative_H)

                # Accumulate motion for stats
                dx = H_smoothed[0, 2]
                dy = H_smoothed[1, 2]
                self._total_motion_pixels += float(np.sqrt(dx**2 + dy**2))

        # Update previous frame state
        self._prev_gray = gray.copy()

        # Prefer tracked points; fall back to fresh detection if too few remain
        if len(matched_curr) >= 20:
            self._prev_keypoints = matched_curr.reshape(-1, 1, 2)
        else:
            # Re-detect features
            keypoints = self._field_detector.detect_field_features(frame)[0]
            if len(keypoints) > 0:
                self._prev_keypoints = keypoints
            else:
                self._prev_keypoints = self._extract_general_features(gray)

        self._frame_count += 1

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _track_features(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Track features between frames using optical flow."""
        if self._prev_keypoints is None or len(self._prev_keypoints) < 10:
            return np.empty((0, 2)), np.empty((0, 2))

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_keypoints, None, **self._lk_params)

        # Filter good matches
        good_prev = self._prev_keypoints[status == 1].reshape(-1, 2)
        good_curr = curr_pts[status == 1].reshape(-1, 2)

        return good_prev, good_curr

    def _estimate_homography(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> Optional[np.ndarray]:
        """Estimate homography between matched feature points."""
        if len(prev_pts) < 4:
            return None

        H, mask = cv2.findHomography(
            prev_pts.reshape(-1, 1, 2).astype(np.float32),
            curr_pts.reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=2000,
            confidence=0.995
        )

        if H is None:
            return None

        # Reject if fewer than 30% of matches are inliers
        if mask is not None and np.sum(mask) / len(mask) < 0.3:
            return None

        return H

    def _smooth_transform(self, H: np.ndarray) -> np.ndarray:
        """Smooth the per-frame homography using a moving-average window."""
        dx = H[0, 2]
        dy = H[1, 2]
        da = np.arctan2(H[1, 0], H[0, 0])

        # Ignore sub-pixel jitter
        if np.sqrt(dx**2 + dy**2) < self.motion_threshold:
            return np.eye(3, dtype=np.float64)

        self._transform_history.append((dx, dy, da))
        if len(self._transform_history) > self.smoothing_window:
            self._transform_history.pop(0)

        avg_dx = float(np.mean([h[0] for h in self._transform_history]))
        avg_dy = float(np.mean([h[1] for h in self._transform_history]))
        avg_da = float(np.mean([h[2] for h in self._transform_history]))

        cos_a, sin_a = np.cos(avg_da), np.sin(avg_da)
        return np.array([
            [cos_a, -sin_a, avg_dx],
            [sin_a,  cos_a, avg_dy],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

    def _extract_general_features(self, gray: np.ndarray) -> np.ndarray:
        """Fallback to general feature extraction when field features are insufficient."""
        keypoints, _ = self._feature_detector.detectAndCompute(gray, None)
        if not keypoints:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.array([[kp.pt] for kp in keypoints], dtype=np.float32)

    def reset(self) -> None:
        """Reset the stabilizer state for a new video."""
        self._prev_gray = None
        self._prev_keypoints = None
        self._frame_count = 0
        self._transform_history.clear()
        self._total_motion_pixels = 0.0
        self._cumulative_H = np.eye(3, dtype=np.float64)
        self._cumulative_inv_H = np.eye(3, dtype=np.float64)


def create_field_camera_stabilizer(**kwargs) -> FieldCameraStabilizer:
    """
    Factory function to create a FieldCameraStabilizer with recommended defaults.
    """
    defaults = {
        'max_features': 200,
        'smoothing_window': 10,
        'ransac_threshold': 3.0,
        'motion_threshold': 1.0,
        'enabled': True
    }
    defaults.update(kwargs)
    return FieldCameraStabilizer(**defaults)