"""
NFL Route Tracker - Motion Tracker Module
==========================================

Implements motion detection using temporal image differencing (frame differencing).

Author: Sam Gold
Phase: 1 - Foundation

The process:
1. Convert frames to grayscale (simplifies comparison)
2. Apply Gaussian blur (reduces noise)
3. Compute absolute difference between frames
4. Threshold the difference (binary: motion or no motion)
5. Apply morphological operations (clean up noise, fill gaps)
6. Find contours (connected regions of motion)
7. Filter contours by size (remove noise)
8. Extract object positions from valid contours
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.core.config import MotionTrackerConfig, DEFAULT_MOTION_CONFIG
from nfl_route_tracker.tracking.trajectory import Detection, Trajectory, TrajectoryStore

@dataclass
class MotionBlob:
    """
    Represents a detected region of motion in a single frame.

    This is the raw output from motion detection before we try to
    associate it with tracked objects across frames.

    Attributes:
    -----------
    x, y : Top-left corner of bounding box
    width, height : Size of bounding box
    centroid : Center of mass of the contour (more precise than box center)
    area : Pixel area of the contour (not the bounding box)
    contour : The actual contour points (for visualization)
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
    Detects and tracks moving objects using temporal frame differencing.

    How to Use:
    -----------
    ```python
    # Create tracker with custom config
    config = MotionTrackerConfig(threshold=30, min_contour_area=150)
    tracker = MotionTracker(config)

    # Process video
    results = tracker.process_video("my_video.mp4")

    # Analyze results
    print(results.get_summary())

    # Save for later
    results.save("tracking_results.json")
    ```
    """

    def __init__(self, config: Optional[MotionTrackerConfig] = None):
        """
        Initialize the motion tracker.

        Parameters:
        -----------
        config : MotionTrackerConfig, Configuration settings. If None, uses defaults.
        """
        self.config = config or DEFAULT_MOTION_CONFIG

        # Store previous frame for differencing
        self._prev_frame: Optional[np.ndarray] = None

        # For simple tracking: associate blobs across frames
        # This is a VERY simple tracker - we'll improve in Phase 2
        self._next_track_id: int = 0
        self._active_tracks: Dict[int, Tuple[float, float]] = {}  # id -> last position

        # ADDING AS A TRIAL, SEPARATE FROM ORIGINAL, DELETE IF BROKEN
        # self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        #     history=500,        # how many frames to build background model from
        #     varThreshold=80,    # sensitivity - higher = less sensitive to change
        #     detectShadows=False # shadows would create extra detections
        # )

        print("[MotionTracker] Initialized")
        print(f"Config: {self.config}")

    # reset tracker after video ends
    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._prev_frame = None
        self._next_track_id = 0
        self._active_tracks = {}
        print("[MotionTracker] State reset")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a frame for motion detection.

        -----------
        frame : Input frame (BGR or grayscale)

        Returns: Preprocessed grayscale frame
        -----------
        """
        # Convert to grayscale if color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            self.config.blur_kernel_size,
            0  # sigmaX=0 means auto-calculate from kernel size
        )

        # remove background
        # blurred = ewjkfnkajwnds

        return blurred

    # comparing pixels over frames
    def _compute_motion_mask(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute binary motion mask between two frames.

        Parameters:
        -----------
        prev_frame : Previous frame (preprocessed)
        curr_frame : Current frame (preprocessed)

        Returns: Binary mask (255 where motion detected, 0 elsewhere)
        -----------
        """

        # TRIAL, DELETE IF BROKEN
        # fg_mask = self._bg_subtractor.apply(curr_frame)
        # kernel = np.ones((5, 5), np.uint8)
        # dilated = cv2.dilate(fg_mask, kernel, iterations=self.config.dilation_iterations)
        # return dilated

        # compute absolute difference
        diff = cv2.absdiff(prev_frame, curr_frame)

        # threshold to create binary mask
        _, thresh = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)

        # apply morphological dilation, fill gaps in motion regions
        kernel = np.ones((5, 5), np.uint8)  
        dilated = cv2.dilate(thresh, kernel, iterations=self.config.dilation_iterations)

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
            #print("Filter Rejected")
            return False

        # Calculate bounding box area
        bbox_area = blob.width * blob.height

        # same check, for the size of bounding box
        if not (self.config.min_area <= bbox_area <= self.config.max_area):
            #print(f"Filter rejected")
            return False

        # All checks passed
        return True

    def _find_motion_blobs(self, motion_mask: np.ndarray) -> List[MotionBlob]:
        """
        Find connected regions of motion in the mask.

        Parameters:
        -----------
        motion_mask : Binary motion mask

        Returns: List of detected motion regions that pass size filtering
        """
        # Find contours (connected components)
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,  # Only outermost contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress contour points
        )

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
            # Moments are weighted averages of pixel positions
            M = cv2.moments(contour)
            if M["m00"] > 0:  # Avoid division by zero
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            blob = MotionBlob(
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                centroid=(cx, cy),
                area=float(area),
                contour=contour
            )

            # Apply size and aspect ratio filtering (NEW)
            if not self._filter_by_size(blob):
                #rejected_count += 1
                continue
            
            blobs.append(blob)

            # sanity check to see how many boxes thrown away
            # if rejected_count > 0:
            #     print(f"Removed {rejected_count} bounding boxes due to size/shape")

        return blobs

    def _associate_blobs_to_tracks(self, blobs: List[MotionBlob], max_distance: float = 50.0) -> List[Tuple[int, MotionBlob]]:
        """
        Associate detected blobs with existing tracks.

        The algorithm:
        1. For each blob, find the nearest existing track
        2. If close enough (< max_distance), assign to that track
        3. Otherwise, create a new track

        Parameters:
        -----------
        blobs : Detected motion blobs in current frame
        max_distance : Maximum distance to consider a match

        Returns: List[Tuple[int, MotionBlob]], List of (track_id, blob) pairs
        """
        associations = []
        used_tracks = set()

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

        Parameters:
        -----------
        frame : Input frame (BGR or grayscale)
        frame_id : Frame number (for tracking)

        Returns: List of detections in this frame, Motion mask (for visualization)
        """
        # Preprocess current frame
        processed = self._preprocess_frame(frame)

        # Handle first frame (no previous frame to compare)
        if self._prev_frame is None:
            self._prev_frame = processed
            print(f"[MotionTracker] Frame {frame_id}: First frame, initializing")
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
            detection = Detection(
                frame_id=frame_id,
                x=blob.x,
                y=blob.y,
                width=blob.width,
                height=blob.height,
                confidence=min(1.0, blob.area / 1000)  # Larger = more confident
            )
            detections.append((track_id, detection))

        # Update previous frame
        self._prev_frame = processed

        # ANNOYING PRINT STATEMENTS UNCOMMENT IF DONT WANT
        # if len(detections) > 0:
        #     print(f"[MotionTracker] Frame {frame_id}: Found {len(detections)} objects")

        return detections, motion_mask

    def process_video(self, video_path: str, max_frames: Optional[int] = None, save_masks: bool = False) -> TrajectoryStore:
        """
        Process an entire video and extract trajectories.

        Parameters:
        -----------
        video_path : Path to video file
        max_frames : Maximum frames to process (for testing)
        save_masks : bool, If True, save motion masks (for debugging)

        Returns: All detected trajectories
        """
        from nfl_route_tracker.core.video_loader import VideoLoader

        print(f"\n[MotionTracker] Starting video processing: {video_path}")
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
                    print(f"[MotionTracker] Reached max_frames limit ({max_frames})")
                    break

                # Process frame
                detections, motion_mask = self.process_frame(frame, frame_id)

                # Store detections
                for track_id, detection in detections:
                    store.add_detection(track_id, detection)

                # Optionally save mask
                if save_masks:
                    masks.append(motion_mask.copy())

                # Progress update
                if frame_id > 0 and frame_id % 50 == 0:
                    print(f"[MotionTracker] Progress: {frame_id}/{total} frames "
                          f"({100*frame_id/total:.1f}%)")

        print("="*60)
        print(f"[MotionTracker] Processing complete!")
        print(store.get_summary())

        if save_masks:
            return store, masks
        return store

# ============================================================================
# Visualization Helpers (for debugging and understanding)
# ============================================================================

def draw_detections(frame: np.ndarray, detections: List[Tuple[int, Detection]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw detection bounding boxes on a frame.

    Parameters:
    -----------
    frame : Frame to draw on (will be copied)
    detections : List of (track_id, detection) pairs
    color : BGR color for boxes

    Returns: Frame with boxes drawn
    """
    output = frame.copy()

    for track_id, det in detections:
        # Draw bounding box
        x, y = int(det.x), int(det.y)
        w, h = int(det.width), int(det.height)
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

        # Draw track ID
        cv2.putText(
            output,
            f"ID:{track_id}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # Draw center point
        cx, cy = det.center
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    return output

# testing 
if __name__ == "__main__":
    """
    Test the MotionTracker module.

    This creates a synthetic video with known motion patterns
    and verifies the tracker can detect them.
    """
    print("\n" + "="*60)
    print("Testing MotionTracker Module")
    print("="*60 + "\n")

    test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
    # change the *.mp4 here and be more specific if only want to test one or a few videos
    video_files = list(test_folder.glob("*.mp4"))

    if not video_files:
        print(f"No video files found in {test_folder}")
    else:
        print("\n" + "="*60)
        print("Testing VideoLoader on existing MP4 files")
        print("="*60 + "\n")
        # if test videos exist, go through each one and test video_loader
        # for video_path in video_files:
        # choosing one video as an example
        video_path = video_files[3]
        print(f"Loading video: {video_path.name}")

        print("Running motion tracker...")
        config = MotionTrackerConfig(
            threshold=25,
            min_contour_area=100,
            blur_kernel_size=(5, 5))
        
        tracker = MotionTracker(config)
        results = tracker.process_video(str(video_path))

        print(f"\nResults: {results.num_trajectories} trajectories found")
        # # We expect 1 trajectory (one moving object)
        # assert results.num_trajectories >= 1, "Expected at least 1 trajectory"
        # print("         PASSED!\n")

        # Verify trajectory makes sense
        print("Verifying trajectory quality...")
        traj = results.get_all_trajectories()[0]
        frames, xs, ys = traj.get_path()

        print(f"Trajectory length: {len(traj)} detections")
        print(f"X range: {xs.min():.1f} to {xs.max():.1f}")
        print(f"Total distance: {traj.get_total_distance():.1f} pixels")

        # # Object should be moving right (X increasing)
        # assert xs[-1] > xs[0], "Expected X to increase (moving right)"
        # Should have many detections
        assert len(traj) > 30, f"Expected >30 detections, got {len(traj)}"
        print("PASSED!\n")

        # testing with masks
        print("Running with motion mask output...")
        tracker.reset()
        results, masks = tracker.process_video(str(video_path), max_frames=30, save_masks=True)
        print(f"Got {len(masks)} motion masks")
        assert len(masks) == 30
        print("PASSED!\n")
