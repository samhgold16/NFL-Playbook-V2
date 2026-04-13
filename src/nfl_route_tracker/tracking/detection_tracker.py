"""
NFL Route Tracker - Detection Tracker Pipeline
===============================================

Unified pipeline combining YOLO detection with DeepSORT tracking.
Uses classes from player_detector.py and object_tracker.py
"""

# important packages
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time
import sys
import cv2
import json

# importing global variables and other functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.core.video_loader import VideoLoader
from nfl_route_tracker.tracking.trajectory import TrajectoryStore, Trajectory, Detection
from nfl_route_tracker.core.config import (
    TrackerConfig, DetectorConfig, DetectionTrackerConfig,
    NFLDetectionFilterConfig, CameraStabilizerConfig,
    FieldCameraStabilizerConfig, FieldOrientationConfig
)
from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.detection.nfl_filter import NFLDetectionFilter
from nfl_route_tracker.tracking.bytetrack_tracker import ByteTrackTracker, Track
from nfl_route_tracker.tracking.camera_stabilizer import CameraStabilizer
from nfl_route_tracker.tracking.field_camera_stabilizer import FieldCameraStabilizer
from nfl_route_tracker.tracking.field_orientation_detector import FieldOrientationDetector
from nfl_route_tracker.tracking.trajectory_merger import TrajectoryMerger, merge_trajectory_store

# main class
class DetectionTracker:
    """
    This class combines YOLO-based player detection with DeepSORT multi-object
    tracking to produce consistent trajectory data from video input.
    ```
    """
    def __init__(self, config: Optional[DetectionTrackerConfig] = None):
        """
        Initialize the detection + tracking pipeline.
        """
        # set DetectorConfig and TrackerConfig attributes or use global
        self.config = config or DetectionTrackerConfig()

        # Initialize detector
        print("Initializing YOLO Detector...")
        self._detector = PlayerDetector(self.config.detector_config)

        # Initialize NFL-specific filter
        print("Initializing NFL Detection Filter...")
        self._nfl_filter = NFLDetectionFilter(self.config.nfl_filter_config)

        # Initialize ByteTrack tracker with YOLO model reference for integrated tracking
        print("\nInitializing ByteTrack Tracker...")
        self._tracker = ByteTrackTracker(self.config.tracker_config, yolo_model = self._detector.model)

        if self.config.field_orientation_config.enabled:
            print("Field orientation enabled - using standalone CameraStabilizer for per-frame motion")
            print("  (Disabling ByteTrack GMC to get raw trajectories for field correction)")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
            self._field_camera_stabilizer = None
            self._stabilization_enabled = self.config.camera_config.enabled
            self._use_bytetrack_gmc = False
        elif self.config.tracker_config.gmc_method and self.config.tracker_config.gmc_method != 'none':
            # Standard mode: Let ByteTrack handle GMC for tracking, we'll handle stabilization separately
            print(f"Using ByteTrack's built-in GMC (method: {self.config.tracker_config.gmc_method})")
            self._camera_stabilizer = None
            self._field_camera_stabilizer = None
            self._stabilization_enabled = False
            self._use_bytetrack_gmc = True
        else:
            # Fallback: use standalone CameraStabilizer
            if self.config.field_camera_config.use_field_stabilizer:
                print("Initializing Field-Specific Camera Stabilizer...")
                self._field_camera_stabilizer = FieldCameraStabilizer(
                    max_features=self.config.field_camera_config.max_features,
                    smoothing_window=self.config.field_camera_config.smoothing_window,
                    ransac_threshold=self.config.field_camera_config.ransac_threshold,
                    motion_threshold=self.config.field_camera_config.motion_threshold,
                    enabled=self.config.field_camera_config.enabled
                )
                self._camera_stabilizer = None
                self._stabilization_enabled = self.config.field_camera_config.enabled
            else:
                print("Initializing Camera Stabilizer...")
                self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
                self._field_camera_stabilizer = None
                self._stabilization_enabled = self.config.camera_config.enabled
            self._use_bytetrack_gmc = False

        # Initialize Trajectory Merger for post-processing fragmented tracks
        print("Initializing Trajectory Merger...")
        self._trajectory_merger = TrajectoryMerger(
            spatial_threshold=self.config.tracker_config.merger_spatial_threshold,
            temporal_threshold=self.config.tracker_config.merger_temporal_threshold,
            confidence_threshold=self.config.tracker_config.merger_confidence_threshold,
            density_radius=self.config.tracker_config.merger_density_radius,
            density_threshold=self.config.tracker_config.merger_density_threshold,
            max_merges=self.config.tracker_config.merger_max_merges
        )
        self._merger_enabled = self.config.tracker_config.enable_trajectory_merging

        # Initialize Field Orientation Detector (for perspective correction)
        # This runs ONCE per video on the first frame to correct camera angle
        if self.config.field_orientation_config.enabled:
            print("Initializing Field Orientation Detector...")
            self._field_detector = FieldOrientationDetector(
                video_width=self.config.field_orientation_config.video_width,
                video_height=self.config.field_orientation_config.video_height,
                canny_low=self.config.field_orientation_config.canny_low,
                canny_high=self.config.field_orientation_config.canny_high,
                hough_threshold=self.config.field_orientation_config.hough_threshold,
                hough_min_line_length=self.config.field_orientation_config.hough_min_line_length,
                hough_max_line_gap=self.config.field_orientation_config.hough_max_line_gap,
                angle_tolerance=self.config.field_orientation_config.angle_tolerance,
                min_field_lines=self.config.field_orientation_config.min_field_lines
            )
        else:
            self._field_detector = None
        self._field_orientation = None

        # Processing statistics
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

    def _filter_detections(self, detections: List[DetectionResult], frame_height: int = 984) -> List[DetectionResult]:
        """
        Filter detections to remove invalid sizes and shapes.
        """
        # Apply NFL filter
        filtered = self._nfl_filter.filter_detections(detections, frame_height)

        # Merge overlapping detections (O-line handling)
        if self.config.nfl_filter_config.merge_overlaps:
            filtered = self._nfl_filter.merge_overlapping_detections(filtered, iou_threshold = self.config.nfl_filter_config.merge_iou_threshold)

        # including bounding box stability code
        filtered = self._nfl_filter.constrain_bbox_area_stability(filtered)

        return filtered

    # code to process an individual frame, use later to iterate through all frames
    # REMOVE PRINT STATEMENTS
    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> List[Track]:
        """
        Process a single frame through the detection + tracking pipeline.
        """
        start_time = time.time()
        frame_height = frame.shape[0]

        # YOLO Detection
        raw_detections = self._detector.detect(frame)
        self._total_detections_raw += len(raw_detections)

        # NFL-specific filtering
        filtered_detections = self._filter_detections(raw_detections, frame_height)
        self._total_detections_filtered += len(filtered_detections)

        # ByteTrack tracking (integrated with YOLO)
        tracks = self._tracker.update(frame, filtered_detections, frame_id)
        self._total_tracks_output += len(tracks)

        if self._stabilization_enabled and self._camera_stabilizer is not None:
            self._camera_stabilizer.update(frame)

            # Get motion stats
            motion_stats = self._camera_stabilizer.get_motion_stats()
            self._total_camera_motion = motion_stats.get('total_motion_pixels', 0)

        # Update field-specific camera stabilizer if enabled
        elif self._stabilization_enabled and self._field_camera_stabilizer is not None:
            self._field_camera_stabilizer.update(frame)

        # Update statistics
        self._total_processing_time += time.time() - start_time
        self._frames_processed += 1

        return tracks
    
    # using process_frame() over an entire video and store data
    # remove filter_short_trajectories or find better way to integrate it if decide to go with it
    def process_video(self, video_path: str, output_video_path: Optional[str] = None,
                      output_json_path: Optional[str] = None, max_frames: Optional[int] = None,
                      filter_short_trajectories: bool = True, filter_off_field: bool = True,
                      field_bounds: Optional[Dict] = None) -> TrajectoryStore:
        """
        Process an entire video file and return trajectory data.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing: {video_path.name}")

        # Reset state for new video
        self._tracker.reset()
        if self._camera_stabilizer is not None:
            self._camera_stabilizer.reset()
        if self._field_detector:
            self._field_detector.reset()
        self._field_orientation = None
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

        # Determine output paths
        if output_video_path is None and self.config.save_video:
            output_video_path = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
        if output_json_path is None and self.config.save_trajectories:
            output_json_path = str(video_path.parent / f"{video_path.stem}_trajectories.json")

        video_writer = None

        # frame loop
        with VideoLoader(str(video_path)) as video:
            total_frames = min(video.metadata.total_frames, max_frames or float('inf'))
            print(f"Video metadata: {video.metadata}")
            print(f"Total frames to process: {int(total_frames)}")

            for frame_id, frame in video:
                if frame_id >= total_frames:
                    break

                # Detect field orientation on FIRST frame only
                # This corrects for the initial camera angle/perspective
                if frame_id == 0 and self._field_detector is not None and self.config.field_orientation_config.enabled:
                    print("\nDetecting field orientation from first frame...")
                    self._field_orientation = self._field_detector.detect_and_compute(frame)
                    print(f"Field orientation detected with confidence: {self._field_orientation.confidence:.2f}")

                # Process frame
                tracks = self.process_frame(frame, frame_id)

                # Write output video if requested
                if output_video_path:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, video.metadata.fps, (w, h))

                    output_frame = self._draw_tracks(frame.copy(), tracks)
                    video_writer.write(output_frame)

                # Progress output
                if self.config.verbose and frame_id % self.config.progress_interval == 0:
                    elapsed = self._total_processing_time
                    fps = self._frames_processed / elapsed if elapsed > 0 else 0
                    #print(f"Processed frame {frame_id}/{int(total_frames)}")
                    print(f"Frame {frame_id}/{int(total_frames)}, Tracks: {len(tracks)}, Total time: {elapsed:.1f}s")

        # Finalize video
        if video_writer:
            video_writer.release()
            print(f"Output video saved: {output_video_path}")

        # POST PROCESSING 

        # Getting and filtering trajectories
        traj_store = self._tracker.get_trajectory_store()

        # Apply trajectory post-processing to merge fragmented tracks
        if self._merger_enabled:
            print("\nApplying trajectory post-processing to merge fragmented tracks...")
            traj_store = self._trajectory_merger.merge_trajectories(traj_store)

        # Apply field orientation correction to transform to orthogonal coordinates
        # This runs ONCE after tracking to correct the camera's initial perspective
        if self._field_detector is not None and self.config.field_orientation_config.enabled:
            traj_store = self._apply_field_orientation_correction(traj_store)

        # stabilizing all trajectories to first frame
        if self._stabilization_enabled and self._camera_stabilizer is not None:
            print("\nApplying standalone camera stabilization to trajectories...")
            traj_store = self._camera_stabilizer.stabilize_trajectory_store(traj_store)
        else:
            print("\nSkipping standalone camera stabilization (using ByteTrack's built-in GMC)")

        # filtering out noise trajs
        if filter_short_trajectories:
            min_length = self.config.tracker_config.min_trajectory_length
            traj_store = self._filter_short_trajectories(traj_store, min_length)

        # Filter off-field trajectories
        if filter_off_field:
            traj_store = self._filter_off_field_trajectories(traj_store, field_bounds, video.metadata if 'video' in dir() else None)

        if output_json_path:
            self._save_trajectories_json(traj_store, output_json_path)

        return traj_store
    
    def _filter_short_trajectories(self, store: TrajectoryStore, min_length: int) -> TrajectoryStore:
        """
        Filter out trajectories that are too short (likely noise).
        """
        filtered_store = TrajectoryStore()
        removed_count = 0

        for traj in store.get_all_trajectories():
            if len(traj) >= min_length:
                for det in traj.detections:
                    filtered_store.add_detection(traj.track_id, det)
            else:
                removed_count += 1

        if removed_count > 0:
            print(f"\nFiltered out {removed_count} short trajectories")

        return filtered_store
    
    def _apply_field_orientation_correction(self, store: TrajectoryStore) -> TrajectoryStore:
        """
        Apply camera motion compensation AND field orientation correction to all trajectory coordinates.

        This handles two issues:
        1. Camera motion (zoom/pan): CameraStabilizer tracked this frame-by-frame, we subtract it
        2. Camera angle (perspective): FieldOrientationDetector computed a one-time rotation

        The order matters:
        - Step 1: For each detection, find cumulative camera transform since frame 0
        - Step 2: Apply inverse camera transform to get position relative to frame 0
        - Step 3: Apply field orientation rotation to orthogonalize coordinates

        Args:
            store: TrajectoryStore with raw trajectories

        Returns:
            TrajectoryStore with corrected coordinates
        """
        if self._field_detector is None:
            print("\nSkipping field correction (no field orientation detector)")
            return store

        print(f"\nApplying camera motion + field orientation correction...")

        # Get field orientation data
        field_homography = self._field_orientation.homography if self._field_orientation else np.eye(3)
        field_angle = self._field_orientation.field_angle if self._field_orientation else 0.0

        print(f"  Field angle: {field_angle:.1f}°")

        corrected_store = TrajectoryStore()

        for traj in store.get_all_trajectories():
            for det in traj.detections:
                x, y = det.center[0], det.center[1]
                frame_id = det.frame_id

                # Step 1: Apply camera motion compensation
                # Get cumulative camera transform from first frame (frame 0) to current frame
                if self._camera_stabilizer is not None and self._stabilization_enabled:
                    camera_transform = self._camera_stabilizer.get_cumulative_transform(frame_id)
                    if camera_transform is not None:
                        # Apply inverse of camera transform to get position as if camera hadn't moved
                        # The camera transform maps frame_0 coordinates to frame_t coordinates
                        # To reverse this, we apply the inverse
                        x, y = self._apply_transform(x, y, camera_transform)

                # Step 2: Apply field orientation rotation
                # Rotate coordinates to make yard lines horizontal
                x, y = self._field_detector.apply_homography(x, y, field_homography)

                # Create corrected detection with new top-left corner
                det_width = det.width
                det_height = det.height
                corrected_det = Detection(
                    frame_id=det.frame_id,
                    x=x - det_width / 2,
                    y=y - det_height / 2,
                    width=det_width,
                    height=det_height,
                    confidence=det.confidence
                )

                corrected_store.add_detection(traj.track_id, corrected_det)

        print(f"  Corrected {store.num_trajectories} trajectories")
        return corrected_store
    
    def _apply_transform(self, x: float, y: float, transform: np.ndarray) -> Tuple[float, float]:
        """
        Apply a 3x3 homography transform to a point.
        """
        point = np.array([x, y, 1.0])
        transformed = transform @ point

        if abs(transformed[2]) > 1e-6:
            return (float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2]))
        else:
            return (float(transformed[0]), float(transformed[1]))
    
    def _filter_off_field_trajectories(self, store: TrajectoryStore, field_bounds: Optional[Dict] = None, video_metadata = None) -> TrajectoryStore:
        """
        Remove trajectories that spend the majority of their time off the field.
        """
        if field_bounds is None:
            if video_metadata is not None:
                # top and bottom 5% y-axis of video being counted
                margin_y = int(video_metadata.height * 0.05)
                field_bounds = {'min_x': 0, 'max_x': video_metadata.width,
                                'min_y': margin_y, 'max_y': video_metadata.height - margin_y}
            else:
                field_bounds = {'min_x': 0, 'max_x': 1920, 'min_y': 50, 'max_y': 934}
 
        filtered_store = TrajectoryStore()
        off_field_count = 0
 
        for trajectory in store.get_all_trajectories():
            total = len(trajectory.detections)
            in_bounds = sum(1 for det in trajectory.detections
                            if (field_bounds['min_x'] <= det.center[0] <= field_bounds['max_x']
                                and field_bounds['min_y'] <= det.center[1] <= field_bounds['max_y']))
 
            if in_bounds / max(1, total) >= 0.25:
                for det in trajectory.detections:
                    filtered_store.add_detection(trajectory.track_id, det)
            else:
                off_field_count += 1
 
        if off_field_count > 0:
            print(f"\nFiltered out {off_field_count} off-field trajectories")
 
        return filtered_store
    
    def _save_trajectories_json(self, store: TrajectoryStore, filepath: str) -> None:
        """
        Save trajectories to JSON format for future route classification tasks.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        avg_detections = (store.total_detections / store.num_trajectories if store.num_trajectories > 0 else 0)

        data = {'metadata': {'num_trajectories': store.num_trajectories,
                            'total_detections': store.total_detections,
                            'average_detections': avg_detections,
                            'frames_processed': self._frames_processed,
                            'processing_time_seconds': self._total_processing_time,
                            'total_camera_motion_pixels': self._total_camera_motion},
                'trajectories': []}

        for traj in store.get_all_trajectories():
            traj_data = {'track_id': traj.track_id,
                        'num_detections': len(traj),
                        'frame_range': traj.get_frame_range(),
                        'total_distance': traj.get_total_distance(),
                        'displacement': traj.get_displacement(),
                        'detections': [{'frame_id': det.frame_id,
                                        'x': float(det.x),
                                        'y': float(det.y),
                                        'width': float(det.width),
                                        'height': float(det.height),
                                        'center_x': float(det.center[0]),
                                        'center_y': float(det.center[1]),
                                        'confidence': float(det.confidence)}
                        for det in traj.detections]}
            data['trajectories'].append(traj_data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Trajectories saved: {filepath}")

    # change here to customize bounding box visualizations
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw track bounding boxes and IDs on a frame.
        """
        import colorsys

        for track in tracks:
            x, y, w, h = track.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Generate color based on track ID
            hue = (int(track.track_id) * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = (int(b * 255), int(g * 255), int(r * 255))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label = f"ID:{track.track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x, y - label_h - 5), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 3), font, font_scale, (255, 255, 255), thickness)

        return frame

    def get_trajectory_store(self) -> TrajectoryStore:
        """Get the trajectory store with all tracked data."""
        return self._tracker.get_trajectory_store()
    
    def get_stabilized_trajectory_store(self) -> TrajectoryStore:
        """Get the stabilized trajectory store (transformed to first-frame coordinates)."""
        if self._camera_stabilizer is None:
            # Using ByteTrack's built-in GMC - trajectories are already stabilized
            return self._tracker.get_trajectory_store()
        raw = self._tracker.get_trajectory_store()
        return self._camera_stabilizer.stabilize_trajectory_store(raw)
    
    def get_trajectories_dataframe(self):
        """Get trajectories as a pandas DataFrame for analysis."""
        return self._tracker.get_trajectory_store().to_dataframe()
    
    def get_camera_motion_stats(self) -> Dict:
        """Get camera motion statistics."""
        if self._camera_stabilizer is None and self._field_camera_stabilizer is None:
            # Using ByteTrack's built-in GMC - stats are handled internally
            return {'using_bytetrack_gmc': True, 'total_motion_pixels': self._total_camera_motion}
        elif self._field_camera_stabilizer is not None:
            return {'using_field_stabilizer': True, 'frame_count': self._field_camera_stabilizer._frame_count}
        return self._camera_stabilizer.get_motion_stats()

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        """
        tracker_stats = self._tracker.get_statistics()
        if self._camera_stabilizer is not None:
            motion_stats = self._camera_stabilizer.get_motion_stats()
        elif self._camera_stabilizer is not None:
            motion_stats = self._camera_stabilizer.get_motion_stats()
        else:
            motion_stats = {'using_bytetrack_gmc': True}

        stats = {'frames_processed': self._frames_processed,
                'total_processing_time': self._total_processing_time,
                'average_fps': (self._frames_processed / self._total_processing_time if self._total_processing_time > 0 else 0),
                'raw_detections': self._total_detections_raw,
                'filtered_detections': self._total_detections_filtered,
                'output_tracks': self._total_tracks_output,
                'filter_ratio': (self._total_detections_filtered / self._total_detections_raw if self._total_detections_raw > 0 else 0),
                'total_camera_motion_pixels': self._total_camera_motion,
                **tracker_stats,}
                #**motion_stats}

        return stats
    
    # RESETTERRRRRRRRRR

    def reset(self):
        """Reset the pipeline state for processing a new video."""
        self._tracker.reset()
        if self._camera_stabilizer is not None:
            self._camera_stabilizer.reset()
        if self._field_camera_stabilizer is not None:
            self._field_camera_stabilizer.reset()
        if self._field_detector is not None:
            self._field_detector.reset()
        self._field_orientation = None
        # Reset NFL filter's area history tracking
        self._nfl_filter.reset_area_history()
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0
        print("Pipeline reset complete.")