"""
NFL Route Tracker - Camera Motion Compensation Module
=====================================================

 Compensates for camera panning/motion in All-22 footage.
"""

# important packages
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import sys

# Import detection result type
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_route_tracker.detection.player_detector import DetectionResult

from nfl_route_tracker.detection.player_detector import DetectionResult
from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore, Trajectory
from nfl_route_tracker.core.config import CameraStabilizerConfig

class CameraStabilizer:
    """
    Compensates for camera motion using homography estimation.
    # finds field attributes in first frame and tracks them, applies homography and stablizes detections
    """
    def __init__(self, config: Optional[CameraStabilizerConfig] = None):
        """
        Initialize the camera stabilizer.
        """
        self.config = config or CameraStabilizerConfig()

        # Feature detector
        if self.config.feature_method == 'orb':
            self._feature_detector = cv2.ORB_create(nfeatures = self.config.max_features, scaleFactor = 1.2, nlevels = 8)
            # matching features across two images with "perspective" changes
            self._feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.config.feature_method == 'sift':
            self._feature_detector = cv2.SIFT_create(nfeatures = self.config.max_features)
            self._feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.config.feature_method == 'shi-tomasi':
            self._feature_detector = None  # Use cv2.goodFeaturesToTrack
        else:
            raise ValueError(f"Unknown feature method: {self.config.feature_method}")

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

        # Per-frame homography H_{t-1 -> t} and its smoothing history
        self._frame_homography: Optional[np.ndarray] = None
        self._homography_history: List[Tuple[float, float, float]] = []

        # Initialised to identity; composed on each update() call.
        self._cumulative_H: np.ndarray = np.eye(3, dtype=np.float64)
        self._cumulative_inv_H: np.ndarray = np.eye(3, dtype=np.float64)

        # Track per-frame cumulative transforms for trajectory correction
        self._frame_transforms: List[Optional[np.ndarray]] = []  # H_{0 -> t} at each frame

        self._frame_count: int = 0
        self._total_motion_pixels: float = 0.0

        # Feature tracking state for LK optical flow parameters
        self._lk_params = dict(winSize = (21, 21), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # =========================================================================
    # simple helper
    # =========================================================================

    def is_ready(self) -> bool:
        """Check if stabilizer has enough frames to compute homography."""
        return self._frame_count > 1

    def get_motion_stats(self) -> dict:
        """Get camera motion statistics."""
        return {'total_motion_pixels': self._total_motion_pixels,
                'avg_motion_per_frame': (self._total_motion_pixels / max(1, self._frame_count - 1)),
                'frames_processed': self._frame_count}

    def get_transform(self) -> Optional[np.ndarray]:
        """Cumulative forward homography H_{0 -> t} (current frame in frame-0 space)."""
        return self._cumulative_H.copy() if self._frame_count > 1 else None

    def get_inverse_transform(self) -> Optional[np.ndarray]:
        """Cumulative inverse homography H_{t -> 0} used for stabilization."""
        return self._cumulative_inv_H.copy() if self._frame_count > 1 else None

    def get_cumulative_transform(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get the cumulative transform H_{0 -> frame_id} for a specific frame.

        This allows correcting trajectory points to their position as if the
        camera hadn't moved since frame 0.

        Args:
            frame_id: The frame number to get the transform for

        Returns:
            3x3 homography matrix or None if not available
        """
        if frame_id < 0 or frame_id >= len(self._frame_transforms):
            return None

        transform = self._frame_transforms[frame_id]
        if transform is None:
            return None

        # Return inverse to map current frame position back to frame 0
        return np.linalg.inv(transform)

    def get_frame_count(self) -> int:
        """Return the number of frames processed."""
        return self._frame_count

    # =========================================================================
    # main coordinate work
    # =========================================================================

    def stabilize_bbox(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """
        Apply the cumulative inverse homography to a bounding box.
        """

        if not self.is_ready():
            return x, y, w, h

        # Four corners as homogeneous column vectors  (3 x 4)

        corners = np.array([[x,     y,     1.0],
                            [x + w, y,     1.0],
                            [x,     y + h, 1.0],
                            [x + w, y + h, 1.0],], dtype = np.float64).T

        transformed = self._cumulative_inv_H @ corners
        transformed /= transformed[2]

        min_x = float(np.min(transformed[0]))
        max_x = float(np.max(transformed[0]))
        min_y = float(np.min(transformed[1]))
        max_y = float(np.max(transformed[1]))

        return min_x, min_y, max_x - min_x, max_y - min_y

    def stabilize_trajectory_detection(self, detection: Detection) -> Detection:
        """
        Stabilize a trajectory Detection (from trajectory.py) to first-frame
        coordinates.
        """
        sx, sy, sw, sh = self.stabilize_bbox(detection.x, detection.y, detection.width, detection.height)
        return Detection(frame_id = detection.frame_id,
                         x = sx, y = sy, width = sw, height = sh,
                         confidence = detection.confidence)

    def stabilize_trajectory_store(self, store: TrajectoryStore) -> TrajectoryStore:
        """
        Stabilize every detection in every trajectory to first-frame coordinates.
        """
        if not self.config.enabled:
            return store

        stabilized_store = TrajectoryStore()

        for trajectory in store.get_all_trajectories():
            for detection in trajectory.detections:
                stabilized_det = self.stabilize_trajectory_detection(detection)
                stabilized_store.add_detection(trajectory.track_id, stabilized_det)

        return stabilized_store

    # =========================================================================
    # main updater
    # =========================================================================

    def update(self, frame: np.ndarray) -> None:
        """
        Update stabilizer with new frame (call after processing each frame), between frames
        """
        if not self.config.enabled:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # First frame: store reference, detect initial features
        if self._frame_count == 0:
            self._prev_gray = gray.copy()
            self._prev_pts = self._detect_features(gray)
            self._frame_count += 1
            # Record identity transform for frame 0
            self._frame_transforms.append(np.eye(3, dtype=np.float64))
            return

        # Subsequent frames: track -> estimate -> compose
        matched_prev, matched_curr, _ = self._track_features(gray)

        if len(matched_prev) >= 4:
            H_frame = self._estimate_homography(matched_prev, matched_curr)

            if H_frame is not None:
                H_smoothed = self._smooth_transform(H_frame)

                # Compose into cumulative homography  H_{0 -> t}
                # Reading right-to-left: first apply the old cumulative
                # transform, then the new per-frame transform.
                self._cumulative_H = H_smoothed @ self._cumulative_H
                self._cumulative_inv_H = np.linalg.inv(self._cumulative_H)

                # Accumulate motion for stats
                dx = H_smoothed[0, 2]
                dy = H_smoothed[1, 2]
                self._total_motion_pixels += float(np.sqrt(dx**2 + dy**2))

        # Update previous-frame state
        self._prev_gray = gray.copy()

        # Prefer tracked points; fall back to fresh detection if too few remain
        if len(matched_curr) >= 20:
            self._prev_pts = matched_curr.reshape(-1, 1, 2)
        else:
            self._prev_pts = self._detect_features(gray)

        self._frame_count += 1

        # Record cumulative transform for this frame
        self._frame_transforms.append(self._cumulative_H.copy())

    # =========================================================================
    # resetter
    # =========================================================================

    def reset(self) -> None:
        """Reset the stabilizer state (call when starting a new video)."""
        self._prev_gray = None
        self._prev_features = None
        self._homography = None
        self._inv_homography = None
        self._frame_count = 0
        self._homography_history = []
        self._total_motion_pixels = 0.0
        self._prev_pts = None
        self._frame_transforms = []

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect features in a grayscale frame.
        """
        if self.config.feature_method == 'shi-tomasi':
            pts = cv2.goodFeaturesToTrack(gray, maxCorners = self.config.max_features,
                                          qualityLevel = self.config.quality_level,
                                          minDistance = int(self.config.min_distance),
                                          blockSize = 7)
            return pts if pts is not None else np.empty((0, 1, 2), dtype = np.float32)

        # ORB / SIFT path
        keypoints, _ = self._feature_detector.detectAndCompute(gray, None)
        if not keypoints:
            return np.empty((0, 1, 2), dtype=np.float32)
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def _track_features(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features using optical flow (Lucas-Kanade).
        """
        if self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_pts = self._detect_features(gray)
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 1))

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None, **self._lk_params)

        good_prev = self._prev_pts[status == 1].reshape(-1, 2)
        good_curr = curr_pts[status == 1].reshape(-1, 2)

        return good_prev, good_curr, status

    def _estimate_homography(self, prev_pts: np.ndarray,
                            curr_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate homography between two sets of matched points.
        """
        if len(prev_pts) < 4:
            return None

        H, mask = cv2.findHomography(prev_pts.reshape(-1, 1, 2).astype(np.float32),
                                    curr_pts.reshape(-1, 1, 2).astype(np.float32),
                                    cv2.RANSAC,
                                    ransacReprojThreshold = self.config.ransac_threshold,
                                    maxIters = 2000,
                                    confidence = 0.995,)

        if H is None:
            return None

        # Reject if fewer than 30 % of matches are inliers
        if mask is not None and np.sum(mask) / len(mask) < 0.3:
            return None

        return H

    def _smooth_transform(self, H: np.ndarray) -> np.ndarray:
        """
        Smooth the per-frame homography using a moving-average window.
        """
        dx = H[0, 2]
        dy = H[1, 2]
        da = np.arctan2(H[1, 0], H[0, 0])

        # Ignore sub-pixel jitter
        if np.sqrt(dx**2 + dy**2) < self.config.motion_threshold:
            return np.eye(3, dtype=np.float64)

        self._homography_history.append((dx, dy, da))
        if len(self._homography_history) > self.config.smoothing_window:
            self._homography_history.pop(0)

        avg_dx = float(np.mean([h[0] for h in self._homography_history]))
        avg_dy = float(np.mean([h[1] for h in self._homography_history]))
        avg_da = float(np.mean([h[2] for h in self._homography_history]))

        cos_a, sin_a = np.cos(avg_da), np.sin(avg_da)
        return np.array([[cos_a, -sin_a, avg_dx],
                         [sin_a,  cos_a, avg_dy],
                         [0.0,    0.0,   1.0]], dtype = np.float64)