#!/usr/bin/env python3
"""
NFL Route Tracker - Detection Tracker V3 with FINAL Fixed Transform
=================================================================

PIPELINE V3: Using FINAL CORRECTED transforms

KEY FIXES FROM V2:
1. Uses FixedFieldOrientationDetector (detects ~80° slant, not ~0°)
2. Uses FinalFieldTransform (rotation-based, correct axis mapping)
3. Players on SAME yardline → similar field_y values
4. Players on DIFFERENT yardlines → different field_y values

COORDINATE SYSTEM:
- Video: X = horizontal (left-right), Y = vertical (top-bottom)
- Field: field_x = width (sideline), field_y = depth (toward end zones)

After FinalFieldTransform:
- rotated_Y = field_depth (same yardline → similar values ✓)
- rotated_X = field_width (sideline position)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time
import sys
import cv2
import json
import matplotlib.pyplot as plt

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
from nfl_route_tracker.tracking.trajectory_merger import TrajectoryMerger

# USE THE FIXED DETECTORS AND TRANSFORMS
from nfl_route_tracker.tracking.fixed_field_orientation_detector import FixedFieldOrientationDetector
from nfl_route_tracker.tracking.final_field_transform import FinalFieldTransform


class DetectionTrackerV3:
    """
    Detection tracker with FINAL CORRECTED field transformation.

    This version uses:
    - FixedFieldOrientationDetector: Correctly detects ~80° yard line slant
    - FinalFieldTransform: Correct rotation-based transform

    Key insight:
    - After -80° rotation, rotated_Y ≈ field_depth (same yardline → similar Y)
    - rotated_X ≈ field_width (sideline position)
    """

    def __init__(self, config: Optional[DetectionTrackerConfig] = None):
        """Initialize the pipeline."""
        self.config = config or DetectionTrackerConfig()

        print("Initializing YOLO Detector...")
        self._detector = PlayerDetector(self.config.detector_config)

        print("Initializing NFL Detection Filter...")
        self._nfl_filter = NFLDetectionFilter(self.config.nfl_filter_config)

        print("\nInitializing ByteTrack Tracker...")
        self._tracker = ByteTrackTracker(self.config.tracker_config, yolo_model=self._detector.model)

        # Camera stabilizer
        if self.config.field_orientation_config.enabled:
            print("Field orientation enabled - using CameraStabilizer for per-frame motion")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
            self._stabilization_enabled = self.config.camera_config.enabled
        else:
            print("Initializing Camera Stabilizer...")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
            self._stabilization_enabled = self.config.camera_config.enabled

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

        # USE FIXED FIELD ORIENTATION DETECTOR
        if self.config.field_orientation_config.enabled:
            print("Initializing FIXED Field Orientation Detector...")
            self._field_detector = FixedFieldOrientationDetector(
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
        self._field_transform = None  # FINAL field transform

        # Statistics
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

    def process_video(self, video_path: str, output_video_path: Optional[str] = None,
                     output_json_path: Optional[str] = None,
                     output_plot_path: Optional[str] = None,
                     max_frames: Optional[int] = None,
                     filter_short_trajectories: bool = True,
                     filter_off_field: bool = True) -> TrajectoryStore:
        """Process video with FINAL corrected field transformation."""

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing: {video_path.name}")

        # Reset state
        self._reset()

        video_writer = None

        with VideoLoader(str(video_path)) as video:
            metadata = video.metadata
            total_frames = min(metadata.total_frames, max_frames or float('inf'))
            print(f"Video: {video_path.name}")
            print(f"Resolution: {metadata.width}x{metadata.height}")
            print(f"Total frames to process: {int(total_frames)}")

            for frame_id, frame in video:
                if frame_id >= total_frames:
                    break

                # Detect field orientation on FIRST frame
                if frame_id == 0 and self._field_detector is not None:
                    print("\n" + "="*60)
                    print("Detecting field orientation (FIXED detector)...")
                    print("="*60)
                    self._field_orientation = self._field_detector.detect_and_compute(frame)
                    print(f"\n  Yard line angle: {self._field_orientation.yard_line_angle:.2f}°")
                    print(f"  Rotation to apply: {self._field_orientation.field_angle:.2f}°")
                    print(f"  Lines found: {self._field_orientation.all_lines_found}")
                    print(f"  Confidence: {self._field_orientation.confidence:.2f}")

                    # Create FINAL field transform
                    if abs(self._field_orientation.yard_line_angle) > 0.5:
                        self._field_transform = FinalFieldTransform(
                            yard_line_angle=self._field_orientation.yard_line_angle,
                            video_width=metadata.width,
                            video_height=metadata.height
                        )
                        print(f"\n  Created FINAL field transform")
                        print(f"  - rotated_Y = field_depth (same yardline → similar values)")
                        print(f"  - rotated_X = field_width (sideline position)")
                    else:
                        print(f"  Minimal slant detected, skipping transform")

                # Process frame
                tracks = self._process_frame(frame, frame_id)

                # Write output video
                if output_video_path:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, metadata.fps, (w, h))
                    output_frame = self._draw_tracks(frame.copy(), tracks)
                    video_writer.write(output_frame)

                # Progress
                if frame_id % 50 == 0:
                    elapsed = self._total_processing_time
                    print(f"Frame {frame_id}/{int(total_frames)}, Tracks: {len(tracks)}, Time: {elapsed:.1f}s")

        if video_writer:
            video_writer.release()
            print(f"\nOutput video saved: {output_video_path}")

        # POST PROCESSING
        traj_store = self._tracker.get_trajectory_store()

        # Merge trajectories
        if self._merger_enabled:
            print("\nMerging fragmented trajectories...")
            traj_store = self._trajectory_merger.merge_trajectories(traj_store)

        # Camera stabilization
        if self._stabilization_enabled and self._camera_stabilizer is not None:
            print("Applying camera stabilization...")
            traj_store = self._camera_stabilizer.stabilize_trajectory_store(traj_store)

        # Apply FINAL field transformation
        if self._field_transform is not None:
            print(f"\nApplying FINAL field transformation...")
            traj_store = self._apply_field_transform(traj_store)

        # Filtering
        if filter_short_trajectories:
            min_length = self.config.tracker_config.min_trajectory_length
            traj_store = self._filter_short_trajectories(traj_store, min_length)

        if filter_off_field:
            traj_store = self._filter_off_field_trajectories(traj_store)

        # Save JSON
        if output_json_path:
            self._save_trajectories_json(traj_store, output_json_path)

        # Generate trajectory plot
        if output_plot_path:
            self._generate_trajectory_plot(traj_store, output_plot_path)

        print(f"\nProcessing complete:")
        print(f"  Frames processed: {self._frames_processed}")
        print(f"  Trajectories: {traj_store.num_trajectories}")
        print(f"  FPS: {self._frames_processed / max(0.1, self._total_processing_time):.2f}")

        return traj_store

    def _reset(self):
        """Reset pipeline state."""
        self._tracker.reset()
        if self._camera_stabilizer:
            self._camera_stabilizer.reset()
        # if self._field_detector:
        #     self._field_detector.reset()
        self._field_orientation = None
        self._field_transform = None
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        self._total_tracks_output = 0
        self._total_camera_motion = 0.0

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> List[Track]:
        """Process a single frame."""
        start_time = time.time()
        frame_height = frame.shape[0]

        raw_detections = self._detector.detect(frame)
        self._total_detections_raw += len(raw_detections)

        filtered = self._nfl_filter.filter_detections(raw_detections, frame_height)
        if self.config.nfl_filter_config.merge_overlaps:
            filtered = self._nfl_filter.merge_overlapping_detections(
                filtered, iou_threshold=self.config.nfl_filter_config.merge_iou_threshold
            )
        filtered = self._nfl_filter.constrain_bbox_area_stability(filtered)
        self._total_detections_filtered += len(filtered)

        tracks = self._tracker.update(frame, filtered, frame_id)
        self._total_tracks_output += len(tracks)

        if self._stabilization_enabled and self._camera_stabilizer:
            self._camera_stabilizer.update(frame)
            motion_stats = self._camera_stabilizer.get_motion_stats()
            self._total_camera_motion = motion_stats.get('total_motion_pixels', 0)

        self._total_processing_time += time.time() - start_time
        self._frames_processed += 1

        return tracks

    def _apply_field_transform(self, store: TrajectoryStore) -> TrajectoryStore:
        """Apply FINAL field transformation to trajectories."""
        if self._field_transform is None:
            return store

        print(f"  Transforming {store.num_trajectories} trajectories...")
        corrected_store = TrajectoryStore()

        for traj in store.get_all_trajectories():
            for det in traj.detections:
                x, y = det.center[0], det.center[1]

                # Apply FINAL field transform
                # Returns (field_x, field_y) where:
                # - field_y = depth (same yardline → similar values)
                # - field_x = width (sideline position)
                field_x, field_y = self._field_transform.transform_point(x, y)

                det_width = det.width
                det_height = det.height
                corrected_det = Detection(
                    frame_id=det.frame_id,
                    x=field_x - det_width / 2,
                    y=field_y - det_height / 2,
                    width=det_width,
                    height=det_height,
                    confidence=det.confidence
                )

                corrected_store.add_detection(traj.track_id, corrected_det)

        print(f"  Transformation complete")
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

    def _filter_off_field_trajectories(self, store: TrajectoryStore) -> TrajectoryStore:
        """Remove off-field trajectories."""
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

        data = {'metadata': {
            'num_trajectories': store.num_trajectories,
            'total_detections': store.total_detections,
            'frames_processed': self._frames_processed,
            'processing_time_seconds': self._total_processing_time,
            'field_transform_applied': self._field_transform is not None,
            'yard_line_angle': self._field_orientation.yard_line_angle if self._field_orientation else 0
        }, 'trajectories': []}

        for traj in store.get_all_trajectories():
            traj_data = {
                'track_id': traj.track_id,
                'num_detections': len(traj),
                'frame_range': traj.get_frame_range(),
                'detections': [{
                    'frame_id': det.frame_id,
                    'x': float(det.x),
                    'y': float(det.y),
                    'center_x': float(det.center[0]),
                    'center_y': float(det.center[1]),
                    'width': float(det.width),
                    'height': float(det.height),
                    'confidence': float(det.confidence)
                } for det in traj.detections]
            }
            data['trajectories'].append(traj_data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nTrajectories saved: {filepath}")

    def _generate_trajectory_plot(self, store: TrajectoryStore, output_path: str) -> None:
        """Generate trajectory visualization plot."""
        print("\nGenerating trajectory plot...")

        fig, ax = plt.subplots(figsize=(16, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        trajectories = store.get_all_trajectories()

        for i, traj in enumerate(trajectories):
            detections = traj.detections
            if len(detections) < 2:
                continue

            xs = [d.center[0] for d in detections]
            ys = [d.center[1] for d in detections]

            color = colors[i % 20]

            # Plot trajectory line
            ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7)

            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=8, zorder=5)
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=8, zorder=5)

            # Label track ID at midpoint
            mid_idx = len(xs) // 2
            ax.annotate(f'{traj.track_id}', (xs[mid_idx], ys[mid_idx]),
                       fontsize=8, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))

        ax.set_xlabel('Field X (Width - sideline position) [pixels]', fontsize=12)
        ax.set_ylabel('Field Y (Depth - toward end zones) [pixels]', fontsize=12)
        ax.set_title(f'Trajectory Plot - {store.num_trajectories} Players\n'
                    f'(Same yardline = similar Y values)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Trajectory plot saved: {output_path}")

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

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        stats = {
            'frames_processed': self._frames_processed,
            'total_processing_time': self._total_processing_time,
            'average_fps': (self._frames_processed / max(0.1, self._total_processing_time)),
            'raw_detections': self._total_detections_raw,
            'filtered_detections': self._total_detections_filtered,
            'output_tracks': self._total_tracks_output,
            'total_camera_motion_pixels': self._total_camera_motion
        }
        return stats


def main():
    """Test the V3 pipeline."""
    import argparse
    from nfl_route_tracker.core.config import get_pipeline_config

    parser = argparse.ArgumentParser(description="NFL Route Tracker V3 - FINAL Fixed Transform")
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', '-o', type=str, default='test_v3_output', help='Output directory')
    parser.add_argument('--max-frames', '-m', type=int, default=500, help='Max frames')
    parser.add_argument('--no-camera-stab', action='store_true', help='Disable camera stabilization')

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  NFL ROUTE TRACKER V3 - FINAL FIXED TRANSFORM")
    print(f"{'='*60}")
    print(f"\nVideo: {args.video}")
    print(f"Output: {output_dir}")
    print(f"\nFIXES APPLIED:")
    print(f"  1. FixedFieldOrientationDetector (detects ~80° slant)")
    print(f"  2. FinalFieldTransform (correct rotation-based transform)")
    print(f"  3. Players on SAME yardline → similar field_y values")

    config = get_pipeline_config()

    # Optionally disable camera stabilization
    if args.no_camera_stab:
        config.camera_config.enabled = False
        print("\nCamera stabilization DISABLED")

    pipeline = DetectionTrackerV3(config)

    video_name = Path(args.video).stem
    output_json = output_dir / f"{video_name}_trajectories.json"
    output_video = output_dir / f"{video_name}_tracked.mp4"
    output_plot = output_dir / f"{video_name}_trajectories.png"

    store = pipeline.process_video(
        args.video,
        output_video_path=str(output_video),
        output_json_path=str(output_json),
        output_plot_path=str(output_plot),
        max_frames=args.max_frames,
        filter_short_trajectories=True,
        filter_off_field=True
    )

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Trajectories: {store.num_trajectories}")
    print(f"Output files:")
    print(f"  Video: {output_video}")
    print(f"  JSON: {output_json}")
    print(f"  Plot: {output_plot}")
    print(f"\nView the trajectory plot to verify:")
    print(f"  - Players on SAME yardline should have SIMILAR Y values")
    print(f"  - Offense/defense should show clear separation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
