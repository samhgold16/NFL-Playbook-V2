"""
NFL Route Tracker - Motion Tracker Module
==========================================

Implements motion detection using temporal image differencing (frame differencing) with openCV.
"""
# important packages
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

# import other repo modules/packages
from nfl_route_tracker.core.config import MotionTrackerConfig, DEFAULT_MOTION_CONFIG
from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore
from nfl_route_tracker.core.video_loader import VideoLoader

@dataclass
class MotionBlob:
    """
    Represents a detected region of motion in a single frame.
    This is the raw output from motion detection of the bounding boxes' attributes.
    """

    x: float
    y: float
    width: float
    height: float
    centroid: Tuple[float, float]
    area: float
    contour: np.ndarray

class MotionTracker:
    """
    Detects and tracks moving objects using temporal pixel frame differencing (opencv).
    """

    def __init__(self, config: Optional[MotionTrackerConfig] = None):
        """
        Initialize the motion tracker with core attributes (or use global set)
        """
        # inheriting from congif file of MotionTrackerConfig()
        self.config = config or DEFAULT_MOTION_CONFIG

        # Store previous frame for differencing
        self._prev_frame: Optional[np.ndarray] = None

        # For simple tracking: associate blobs across frames
        self._next_track_id: int = 0
        self._active_tracks: Dict[int, Tuple[float, float]] = {} 

        print(f"Model Config: {self.config}")

    # reset tracker after video ends
    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._prev_frame = None
        self._next_track_id = 0
        self._active_tracks = {}

    # this deals with greyscale representations, can we incorporate color?
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a frame for motion detection (hue/saturation, gaussian blur).
        -----------
        """
        # Convert to grayscale if color
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hs_only = hsv[:, :, :2]
        else:
            # greyscale fallback 
            hs_only = np.stack([frame, frame], axis=-1)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hs_only, self.config.blur_kernel_size, 0)

        # this would be the place to insert code to remove background potentially
        # blurred = ewjkfnkajwnds

        return blurred

    # comparing pixels over individual frames
    def _compute_motion_mask(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute binary motion mask between two frames.
        """

        # compute absolute difference
        diff = cv2.absdiff(prev_frame, curr_frame)

        # collapse to single channelf or .threshold
        if len(diff.shape) == 3:
            diff = np.max(diff, axis=2).astype(np.uint8)

        # threshold to create binary mask, (min_threshold, 255)
        _, thresh = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)

        # apply morphological dilation, fill gaps in motion regions
        kernel = np.ones((5, 5), np.uint8)  
        # use dilation val (odd) from config.py
        dilated = cv2.dilate(thresh, kernel, iterations = self.config.dilation_iterations)

        return dilated

    # adding another function here to define boundary box size requirements
    def _filter_by_size(self, blob: MotionBlob) -> bool:
        """
        Filter detections based on expected player size and aspect ratio.
        """
        # calculate aspect ratio (width / height) for given area
        if blob.height > 0:
            aspect_ratio = blob.width / blob.height
        else:
            return False

        # Check aspect ratio bounds, reject if outside ratio (set in config file)
        if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
            return False

        # Calculate bounding box area
        bbox_area = blob.width * blob.height

        # same check, for the size of bounding box
        if not (self.config.min_area <= bbox_area <= self.config.max_area):
            return False

        # All checks passed
        return True

    # using motion max from above, using opencv contour 
    def _find_motion_blobs(self, motion_mask: np.ndarray) -> List[MotionBlob]:
        """
        Find connected regions of motion in the mask.
        """
        # Find contours (connected components)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        blobs = []
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)

            # Filter by minimum area (skip noise)
            if area < self.config.min_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M["m00"] > 0: 
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            # assign attributes to class
            blob = MotionBlob(x = float(x), y = float(y),
                              width = float(w), height = float(h),
                              centroid = (cx, cy),area = float(area),
                              contour = contour)

            # apply bounding box size and aspect ratio filtering
            if not self._filter_by_size(blob):
                #rejected_count += 1
                continue
            
            blobs.append(blob)

        return blobs

    # change max_distance value for tracking object permanence?
    # trial and error values
    def _associate_blobs_to_tracks(self, blobs: List[MotionBlob]) -> List[Tuple[int, MotionBlob]]:
        """
        Associate detected blobs with existing tracks if relevant.
        Otherwise initialize a new track from scratch.
        """
        associations = []
        used_tracks = set()
        
        # setting global param
        max_distance = self.config.max_tracking_distance

        # iterate through each mask/blob
        for blob in blobs:
            best_track_id = None
            best_distance = float('inf')

            # Find nearest existing track
            for track_id, last_pos in self._active_tracks.items():
                if track_id in used_tracks:
                    continue

                # Euclidean distance
                dist = np.sqrt((blob.centroid[0] - last_pos[0])**2 + (blob.centroid[1] - last_pos[1])**2)

                if dist < best_distance and dist < max_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                # Match found - use existing track
                associations.append((best_track_id, blob))
                used_tracks.add(best_track_id)
                self._active_tracks[best_track_id] = blob.centroid
            else:
                # No match - create new track
                new_id = self._next_track_id
                self._next_track_id += 1
                associations.append((new_id, blob))
                self._active_tracks[new_id] = blob.centroid
                # print(f"[MotionTracker] Created new track {new_id} at {blob.centroid}")

        return associations

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[List[Detection], np.ndarray]:
        """
        Process a single frame and return detections.
        """
        # Preprocess current frame
        processed = self._preprocess_frame(frame)

        # Handle first frame (no previous frame to compare)
        if self._prev_frame is None:
            self._prev_frame = processed
            print("Initializing first frame...")
            return [], np.zeros_like(processed)

        # Compute motion mask
        motion_mask = self._compute_motion_mask(self._prev_frame, processed)

        # Find motion blobs
        blobs = self._find_motion_blobs(motion_mask)

        # Associate blobs with tracks
        associations = self._associate_blobs_to_tracks(blobs)

        # Convert to Detection objects
        detections = []
        for track_id, blob in associations:
            detection = Detection(frame_id = frame_id, x = blob.x, y = blob.y,
                width = blob.width, height = blob.height,
                confidence = min(1.0, blob.area / 1000))
            detections.append((track_id, detection))

        # Update previous frame
        self._prev_frame = processed

        return detections, motion_mask

    def process_video(self, video_path: str, max_frames: Optional[int] = None, save_masks: bool = False) -> TrajectoryStore:
        """
        Process an entire video and extract trajectories.
        By iteratively going through each frame.
        """

        print(f"\nStarting video processing: {video_path}")
        print("="*60)

        # Reset state for new video
        self.reset()

        # Create trajectory store
        store = TrajectoryStore()

        # Storage for motion masks (optional)
        masks = [] if save_masks else None

        # Process video
        with VideoLoader(video_path, grayscale=False) as video:
            fps = video.metadata.fps
            total = video.metadata.total_frames

            for frame_id, frame in video:
                # Check max frames limit
                if max_frames and frame_id >= max_frames:
                    # print(f"[MotionTracker] Reached max_frames limit ({max_frames})")
                    break

                # Process frame
                detections, motion_mask = self.process_frame(frame, frame_id)

                # Store detections
                for track_id, detection in detections:
                    store.add_detection(track_id, detection)

                # Optionally save mask
                if save_masks:
                    masks.append(motion_mask.copy())

        print("="*60)
        print(f"Processing complete!")

        if save_masks:
            return store, masks
        return store

# ============================================================================
# Visualization Helpers (for debugging and understanding)
# ============================================================================

def draw_detections(frame: np.ndarray, detections: List[Tuple[int, Detection]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw detection bounding boxes on a frame.
    """
    output = frame.copy()

    for track_id, det in detections:
        # Draw bounding box
        x, y = int(det.x), int(det.y)
        w, h = int(det.width), int(det.height)
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

        # Draw track ID
        cv2.putText(output, f"ID:{track_id}", (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw center point
        cx, cy = det.center
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    return output