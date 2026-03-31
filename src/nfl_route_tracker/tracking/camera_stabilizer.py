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

# Import CameraStabilizerConfig from centralized config
from ..core.config import CameraStabilizerConfig

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
            self._feature_detector = cv2.ORB_create(nfeatures = self.config.max_features,
                                                    scaleFactor = 1.2,
                                                    nlevels = 8)
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
        self._prev_gray = None
        self._prev_features = None
        self._homography = None
        self._inv_homography = None
        self._frame_count = 0
        self._homography_history = []
        self._total_motion_pixels = 0.0

        # Feature tracking state
        self._prev_pts = None
        self._lk_params = dict(winSize = (21, 21),
                                maxLevel = 3,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
    def is_ready(self) -> bool:
        """Check if stabilizer has enough frames to compute homography."""
        return self._frame_count > 1 and self._homography is not None

    def get_motion_stats(self) -> dict:
        """Get camera motion statistics."""
        return {'total_motion_pixels': self._total_motion_pixels,
                'avg_motion_per_frame': (self._total_motion_pixels / max(1, self._frame_count - 1)),
                'homography_history_len': len(self._homography_history)}
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect features in a grayscale frame.
        """
        if self.config.feature_method == 'shi-tomasi':
            features = cv2.goodFeaturesToTrack(gray, maxCorners = self.config.max_features,
                                               qualityLevel = self.config.quality_level, 
                                               minDistance = int(self.config.min_distance),
                                               blockSize = 7)
        else:
            # ORB or SIFT
            keypoints, descriptors = self._feature_detector.detectAndCompute(gray, None)
            if keypoints is None or len(keypoints) == 0:
                return np.array([]).reshape(0, 1, 2)

            # Convert keypoints to numpy array
            features = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            features = features.reshape(-1, 1, 2)

        return features
    
    def _track_features(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features using optical flow (Lucas-Kanade).
        """
        if self._prev_pts is None or len(self._prev_pts) < 10:
            # Not enough features, detect new ones
            self._prev_pts = self._detect_features(gray)
            self._prev_gray = gray.copy()
            return np.array([]), np.array([]), np.array([])

        # Track features with optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, 
                                                         self._prev_pts,
                                                         None,
                                                         **self._lk_params)

        # Filter out failed matches
        good_prev = self._prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        return good_prev, good_curr, status

    def _estimate_homography(self, prev_pts: np.ndarray,
                            curr_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate homography between two sets of matched points.
        """
        if len(prev_pts) < 4:
            return None

        # Convert to expected format (N, 1, 2)
        prev_pts_h = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        curr_pts_h = curr_pts.reshape(-1, 1, 2).astype(np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(prev_pts_h, curr_pts_h,
                                     cv2.RANSAC,
                                     ransacReprojThreshold = self.config.ransac_threshold,
                                     maxIters = 2000,
                                     confidence = 0.995)

        if H is None:
            return None

        # Compute inlier ratio to validate homography quality
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            if inlier_ratio < 0.3:
                # Less than 30% inliers - likely failed estimation
                return None

        return H
    
    def _compute_camera_motion(self, H: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract camera motion parameters from homography.
        """
        # Translation (top row of homography gives us scale and x-translation)
        dx = H[0, 2]
        dy = H[1, 2]

        # Rotation angle from rotation matrix components
        # For small rotations: sin(theta) ≈ H[1,0], cos(theta) ≈ H[0,0]
        da = np.arctan2(H[1, 0], H[0, 0])

        return dx, dy, da
    
    def _smooth_transform(self, H: np.ndarray) -> np.ndarray:
        """
        Smooth homography using moving average of recent transforms.
        """
        # Extract motion parameters
        dx, dy, da = self._compute_camera_motion(H)

        # Check if motion is below threshold (no significant camera motion)
        motion_magnitude = np.sqrt(dx**2 + dy**2)
        if motion_magnitude < self.config.motion_threshold:
            # Return identity transform
            return np.eye(3)

        # Add to history
        self._homography_history.append((dx, dy, da))
        if len(self._homography_history) > self.config.smoothing_window:
            self._homography_history.pop(0)

        # Compute smoothed parameters
        avg_dx = np.mean([h[0] for h in self._homography_history])
        avg_dy = np.mean([h[1] for h in self._homography_history])
        avg_da = np.mean([h[2] for h in self._homography_history])

        # Rebuild homography from smoothed parameters
        cos_a = np.cos(avg_da)
        sin_a = np.sin(avg_da)

        H_smoothed = np.array([[cos_a, -sin_a, avg_dx],
                               [sin_a,  cos_a, avg_dy],
                               [0.0,    0.0,   1.0]], dtype = np.float64)

        return H_smoothed
    
    def update(self, frame: np.ndarray) -> None:
        """
        Update stabilizer with new frame (call after processing each frame), between frames
        """
        if not self.config.enabled:
            return

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self._frame_count == 0:
            # First frame - just store it
            self._prev_gray = gray.copy()
            self._prev_pts = self._detect_features(gray)
            self._frame_count += 1
            return

        # Track features between frames
        matched_prev, matched_curr, status = self._track_features(gray)

        if len(matched_prev) >= 4:
            # Estimate homography
            H = self._estimate_homography(matched_prev, matched_curr)

            if H is not None:
                # Smooth the transform
                H_smoothed = self._smooth_transform(H)

                # Store for stabilization
                self._homography = H_smoothed
                self._inv_homography = np.linalg.inv(H_smoothed)

                # Track total motion
                dx, dy, da = self._compute_camera_motion(H_smoothed)
                self._total_motion_pixels += np.sqrt(dx**2 + dy**2)

        # Update previous frame and features
        self._prev_gray = gray.copy()

        # Keep tracked features or detect new ones
        if len(matched_curr) >= 20:
            self._prev_pts = matched_curr.reshape(-1, 1, 2)
        else:
            self._prev_pts = self._detect_features(gray)

        self._frame_count += 1

    def stabilize_detection(self, detection: DetectionResult) -> DetectionResult:
        """
        Apply inverse homography to a single detection to stabilize it.

        This transforms the bounding box from the current frame's coordinate
        system back to the reference frame (first frame), compensating for
        camera motion.
        """
        if not self.is_ready() or self._inv_homography is None:
            return detection

        # Get detection corners (x, y)
        x, y = detection.x, detection.y
        w, h = detection.width, detection.height

        # Get center point
        cx = x + w / 2
        cy = y + h / 2

        # Apply inverse homography to center point
        pt = np.array([cx, cy, 1.0])
        transformed = self._inv_homography @ pt
        transformed = transformed / transformed[2]  # Normalize

        # Compute transformed width/height (approximate)
        # For small transforms, scaling is roughly constant
        corners = np.array([[x, y, 1],
                            [x + w, y, 1],
                            [x, y + h, 1],
                            [x + w, y + h, 1]]).T

        transformed_corners = self._inv_homography @ corners
        transformed_corners = transformed_corners / transformed_corners[2]

        min_x = np.min(transformed_corners[0])
        max_x = np.max(transformed_corners[0])
        min_y = np.min(transformed_corners[1])
        max_y = np.max(transformed_corners[1])

        new_x = min_x
        new_y = min_y
        new_w = max_x - min_x
        new_h = max_y - min_y

        # Create stabilized detection
        return DetectionResult(x = new_x, y = new_y,
                               width = new_w, height = new_h,
                               confidence = detection.confidence,
                               class_id = detection.class_id,
                               class_name = detection.class_name)
    
    def stabilize_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Stabilize all detections by applying inverse homography.
        """
        if not self.is_ready():
            return detections

        return [self.stabilize_detection(det) for det in detections]
    
    def get_inverse_transform(self) -> Optional[np.ndarray]:
        """
        Get the current inverse homography matrix.
        """
        return self._inv_homography
    
    def get_transform(self) -> Optional[np.ndarray]:
        """
        Get the current forward homography matrix.
        """
        return self._homography

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


# utility functions to visualize homography and motion (for debugging)
def visualize_stabilization(frame: np.ndarray,
                            detections: List[DetectionResult],
                            stabilized_detections: List[DetectionResult]) -> np.ndarray:
    """
    Create side-by-side visualization of original vs stabilized detections.
    """
    # Draw original detections
    frame_orig = frame.copy()
    for det in detections:
        x, y, w, h = int(det.x), int(det.y), int(det.width), int(det.height)
        cv2.rectangle(frame_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_orig, f"({det.x:.0f},{det.y:.0f})", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw stabilized detections
    frame_stab = frame.copy()
    for det in stabilized_detections:
        x, y, w, h = int(det.x), int(det.y), int(det.width), int(det.height)
        cv2.rectangle(frame_stab, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_stab, f"({det.x:.0f},{det.y:.0f})", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Concatenate side by side
    combined = np.hstack([frame_orig, frame_stab])

    # Add labels
    cv2.putText(combined, "ORIGINAL", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "STABILIZED", (frame.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return combined
