#!/usr/bin/env python3
"""
NFL Route Tracker - Detection Tracker with Field Axis Projection
================================================================

This is a patched version of detection_tracker.py that integrates the
FieldAxisProjectionTransform to fix the spatial relationship preservation issue.

Changes from original:
1. Uses FieldAxisProjectionTransform instead of simple rotation homography
2. Ensures players on same yardline have similar X coordinates after transformation
3. Better handles the slanted yardline issue in All-22 footage
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time
import sys
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nfl_route_tracker.core.video_loader import VideoLoader
from nfl_route_tracker.tracking.trajectory import TrajectoryStore, Trajectory, Detection
from nfl_route_tracker.core.config import (
    TrackerConfig, DetectorConfig, DetectionTrackerConfig,
    NFLDetectionFilterConfig, CameraStabilizerConfig,
    FieldOrientationConfig
)
from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.detection.nfl_filter import NFLDetectionFilter
from nfl_route_tracker.tracking.bytetrack_tracker import ByteTrackTracker, Track
from nfl_route_tracker.tracking.camera_stabilizer import CameraStabilizer
from nfl_route_tracker.tracking.field_orientation_detector import FieldOrientationDetector
from nfl_route_tracker.tracking.trajectory_merger import TrajectoryMerger, merge_trajectory_store
from nfl_route_tracker.tracking.field_axis_transform import FieldAxisProjectionTransform


class DetectionTrackerWithAxisProjection:
    """
    Detection tracker with improved field axis projection.

    This version replaces the simple rotation with FieldAxisProjectionTransform
    to preserve spatial relationships between players on the same yardline.
    """

    def __init__(self, config: Optional[DetectionTrackerConfig] = None):
        """Initialize the pipeline."""
        self.config = config or DetectionTrackerConfig()

        # Initialize components
        print("Initializing YOLO Detector...")
        self._detector = PlayerDetector(self.config.detector_config)

        print("Initializing NFL Detection Filter...")
        self._nfl_filter = NFLDetectionFilter(self.config.nfl_filter_config)

        print("\nInitializing ByteTrack Tracker...")
        self._tracker = ByteTrackTracker(self.config.tracker_config, yolo_model=self._detector.model)

        # Camera stabilizer
        if self.config.field_orientation_config.enabled:
            print("Field orientation enabled - using CameraStabilizer for per-frame motion")
            print("  (ByteTrack GMC disabled for proper field correction)")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
            self._stabilization_enabled = self.config.camera_config.enabled
            self._use_bytetrack_gmc = False
        else:
            print("Initializing Camera Stabilizer...")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
            self._stabilization_enabled = self.config.camera_config.enabled
            self._use_bytetrack_gmc = False

        # Trajectory merger
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

        # Field orientation detector
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
        self._field_axis_transform = None  # NEW: Field axis projection transform

        # Statistics
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

    def _filter_detections(self, detections: List[DetectionResult], frame_height: int = 984) -> List[DetectionResult]:
        """Filter detections."""
        filtered = self._nfl_filter.filter_detections(detections, frame_height)
        if self.config.nfl_filter_config.merge_overlaps:
            filtered = self._nfl_filter.merge_overlapping_detections(filtered, iou_threshold=self.config.nfl_filter_config.merge_iou_threshold)
        filtered = self._nfl_filter.constrain_bbox_area_stability(filtered)
        return filtered

    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> List[Track]:
        """Process a single frame."""
        start_time = time.time()
        frame_height = frame.shape[0]

        # Detection
        raw_detections = self._detector.detect(frame)
        self._total_detections_raw += len(raw_detections)

        # Filtering
        filtered_detections = self._filter_detections(raw_detections, frame_height)
        self._total_detections_filtered += len(filtered_detections)

        # Tracking
        tracks = self._tracker.update(frame, filtered_detections, frame_id)
        self._total_tracks_output += len(tracks)

        # Camera stabilizer
        if self._stabilization_enabled and self._camera_stabilizer is not None:
            self._camera_stabilizer.update(frame)
            motion_stats = self._camera_stabilizer.get_motion_stats()
            self._total_camera_motion = motion_stats.get('total_motion_pixels', 0)

        self._total_processing_time += time.time() - start_time
        self._frames_processed += 1

        return tracks

    def process_video(self, video_path: str, output_video_path: Optional[str] = None,
                     output_json_path: Optional[str] = None, max_frames: Optional[int] = None,
                     filter_short_trajectories: bool = True, filter_off_field: bool = True,
                     field_bounds: Optional[Dict] = None) -> TrajectoryStore:
        """Process video with field axis projection correction."""

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing: {video_path.name}")

        # Reset state
        self._tracker.reset()
        if self._camera_stabilizer:
            self._camera_stabilizer.reset()
        if self._field_detector:
            self._field_detector.reset()
        self._field_orientation = None
        self._field_axis_transform = None
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

        video_writer = None

        with VideoLoader(str(video_path)) as video:
            total_frames = min(video.metadata.total_frames, max_frames or float('inf'))
            print(f"Video metadata: {video.metadata}")
            print(f"Total frames to process: {int(total_frames)}")

            for frame_id, frame in video:
                if frame_id >= total_frames:
                    break

                # Detect field orientation on FIRST frame only
                if frame_id == 0 and self._field_detector is not None and self.config.field_orientation_config.enabled:
                    print("\nDetecting field orientation from first frame...")
                    self._field_orientation = self._field_detector.detect_and_compute(frame)
                    print(f"Field orientation detected with confidence: {self._field_orientation.confidence:.2f}")

                    # NEW: Create field axis projection transform
                    if self._field_orientation.field_angle != 0:
                        self._field_axis_transform = FieldAxisProjectionTransform(
                            yard_line_angle=self._field_orientation.yard_line_angle,
                            center_x=video.metadata.width / 2,
                            center_y=video.metadata.height / 2
                        )
                        print(f"Created Field Axis Projection Transform for angle: {self._field_orientation.yard_line_angle:.2f}°")
                    else:
                        self._field_axis_transform = None

                # Process frame
                tracks = self.process_frame(frame, frame_id)

                if output_video_path:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, video.metadata.fps, (w, h))
                    output_frame = self._draw_tracks(frame.copy(), tracks)
                    video_writer.write(output_frame)

                if self.config.verbose and frame_id % self.config.progress_interval == 0:
                    elapsed = self._total_processing_time
                    print(f"Frame {frame_id}/{int(total_frames)}, Tracks: {len(tracks)}, Total time: {elapsed:.1f}s")

        if video_writer:
            video_writer.release()
            print(f"Output video saved: {output_video_path}")

        # POST PROCESSING
        traj_store = self._tracker.get_trajectory_store()

        # Step 1: Merge fragmented trajectories
        if self._merger_enabled:
            print("\nApplying trajectory post-processing to merge fragmented tracks...")
            traj_store = self._trajectory_merger.merge_trajectories(traj_store)

        # Step 2: Apply camera stabilization
        if self._stabilization_enabled and self._camera_stabilizer is not None:
            print("\nApplying camera stabilization to trajectories...")
            traj_store = self._camera_stabilizer.stabilize_trajectory_store(traj_store)
        else:
            print("\nSkipping camera stabilization")

        # Step 3: Apply FIELD AXIS PROJECTION (instead of simple rotation)
        if self._field_detector is not None and self.config.field_orientation_config.enabled:
            print("\nApplying field axis projection...")
            traj_store = self._apply_field_axis_projection(traj_store)

        # Filtering
        if filter_short_trajectories:
            min_length = self.config.tracker_config.min_trajectory_length
            traj_store = self._filter_short_trajectories(traj_store, min_length)

        if filter_off_field:
            traj_store = self._filter_off_field_trajectories(traj_store, field_bounds, video.metadata if 'video' in dir() else None)

        if output_json_path:
            self._save_trajectories_json(traj_store, output_json_path)

        return traj_store

    def _apply_field_axis_projection(self, store: TrajectoryStore) -> TrajectoryStore:
        """
        Apply field axis projection to all trajectory coordinates.

        KEY DIFFERENCE FROM SIMPLE ROTATION:
        - Uses axis projection to ensure players on same yardline have similar X
        - Projects points onto field axes (perpendicular/parallel to yard lines)
        - Preserves spatial relationships
        """
        if self._field_axis_transform is None:
            print("  Skipping field transformation (no axis transform available)")
            return store

        print(f"  Field axis projection for angle: {self._field_orientation.yard_line_angle:.2f}°")

        corrected_store = TrajectoryStore()

        for traj in store.get_all_trajectories():
            for det in traj.detections:
                x, y = det.center[0], det.center[1]

                # Use field axis projection instead of rotation
                x, y = self._field_axis_transform.transform_point(x, y)

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

        print(f"  Corrected {store.num_trajectories} trajectories using field axis projection")
        return corrected_store

    def _filter_short_trajectories(self, store: TrajectoryStore, min_length: int) -> TrajectoryStore:
        """Filter out short trajectories."""
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

    def _filter_off_field_trajectories(self, store: TrajectoryStore, field_bounds: Optional[Dict] = None,
                                       video_metadata=None) -> TrajectoryStore:
        """Remove off-field trajectories."""
        if field_bounds is None:
            if video_metadata is not None:
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
        """Save trajectories to JSON."""
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

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw track bounding boxes and IDs."""
        import colorsys

        for track in tracks:
            x, y, w, h = track.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            hue = (int(track.track_id) * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = (int(b * 255), int(g * 255), int(r * 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"ID:{track.track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x, y - label_h - 5), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 3), font, font_scale, (255, 255, 255), thickness)

        return frame

    def get_trajectory_store(self) -> TrajectoryStore:
        """Get trajectory store."""
        return self._tracker.get_trajectory_store()

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        tracker_stats = self._tracker.get_statistics()

        if self._camera_stabilizer is not None:
            motion_stats = self._camera_stabilizer.get_motion_stats()
        else:
            motion_stats = {'camera_stabilizer_disabled': True}

        stats = {'frames_processed': self._frames_processed,
                'total_processing_time': self._total_processing_time,
                'average_fps': (self._frames_processed / self._total_processing_time if self._total_processing_time > 0 else 0),
                'raw_detections': self._total_detections_raw,
                'filtered_detections': self._total_detections_filtered,
                'output_tracks': self._total_tracks_output,
                'filter_ratio': (self._total_detections_filtered / self._total_detections_raw if self._total_detections_raw > 0 else 0),
                'total_camera_motion_pixels': self._total_camera_motion,
                **tracker_stats,
                **motion_stats}

        return stats

    def reset(self):
        """Reset pipeline state."""
        self._tracker.reset()
        if self._camera_stabilizer is not None:
            self._camera_stabilizer.reset()
        if self._field_detector is not None:
            self._field_detector.reset()
        self._field_orientation = None
        self._field_axis_transform = None
        self._nfl_filter.reset_area_history()
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0
        print("Pipeline reset complete.")


def main():
    """Test the new pipeline with field axis projection."""
    import argparse
    from nfl_route_tracker.core.config import get_pipeline_config

    parser = argparse.ArgumentParser(description="Test pipeline with field axis projection")
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', '-o', type=str, default='test_axis_projection', help='Output directory')
    parser.add_argument('--max-frames', '-m', type=int, default=500, help='Max frames to process')

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  TESTING FIELD AXIS PROJECTION PIPELINE")
    print(f"{'='*60}")
    print(f"\nVideo: {args.video}")
    print(f"Output: {output_dir}")

    config = get_pipeline_config()
    pipeline = DetectionTrackerWithAxisProjection(config)

    output_json = output_dir / f"{Path(args.video).stem}_trajectories.json"
    output_video = output_dir / f"{Path(args.video).stem}_tracked.mp4"

    store = pipeline.process_video(
        args.video,
        output_video_path=str(output_video),
        output_json_path=str(output_json),
        max_frames=args.max_frames,
        filter_short_trajectories=True,
        filter_off_field=True
    )

    print(f"\nProcessed {pipeline._frames_processed} frames")
    print(f"Output trajectories: {store.num_trajectories}")

    stats = pipeline.get_statistics()
    print(f"\nProcessing FPS: {stats['average_fps']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())