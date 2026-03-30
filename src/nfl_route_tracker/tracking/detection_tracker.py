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
from nfl_route_tracker.tracking.trajectory import TrajectoryStore
from nfl_route_tracker.core.config import (
    TrackerConfig, DetectorConfig, DetectionTrackerConfig,
    NFLDetectionFilterConfig, TemporalAggregatorConfig,
    CameraStabilizerConfig
)
from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.tracking.object_tracker import ObjectTracker, Track
from nfl_route_tracker.detection.nfl_filter import NFLDetectionFilter
from nfl_route_tracker.tracking.temporal_aggregator import TemporalAggregator
from nfl_route_tracker.tracking.camera_stabilizer import CameraStabilizer

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
        self._nfl_filter = NFLDetectionFilter(
            min_area=self.config.nfl_filter_config.min_area,
            max_area=self.config.nfl_filter_config.max_area,
            min_aspect_ratio=self.config.nfl_filter_config.min_aspect_ratio,
            max_aspect_ratio=self.config.nfl_filter_config.max_aspect_ratio,
            min_confidence=self.config.nfl_filter_config.min_confidence,
            low_confidence=self.config.nfl_filter_config.low_confidence,
            near_y_threshold=self.config.nfl_filter_config.near_y_threshold,
            far_y_threshold=self.config.nfl_filter_config.far_y_threshold,
            near_area_range=self.config.nfl_filter_config.near_area_range,
            near_aspect_range=self.config.nfl_filter_config.near_aspect_range,
            far_area_range=self.config.nfl_filter_config.far_area_range,
            far_aspect_range=self.config.nfl_filter_config.far_aspect_range,
            mid_area_range=self.config.nfl_filter_config.mid_area_range,
            mid_aspect_range=self.config.nfl_filter_config.mid_aspect_range,
            min_y_position=self.config.nfl_filter_config.min_y_position,
            max_y_position=self.config.nfl_filter_config.max_y_position
        )

        # Initialize temporal aggregator
        print("Initializing Temporal Aggregator...")
        self._temporal = TemporalAggregator(
            window_size=self.config.temporal_config.window_size,
            stride=self.config.temporal_config.stride,
            method=self.config.temporal_config.aggregation_method,  # Note: uses 'method' in TemporalAggregator
            confidence_weight=self.config.temporal_config.confidence_weight,
            min_detection_count=self.config.temporal_config.min_detection_count,
            position_threshold=self.config.temporal_config.position_threshold
        )

        # Initialize camera stabilizer
        if self.config.camera_config.enabled:
            print("Initializing Camera Stabilizer...")
            self._camera_stabilizer = CameraStabilizer(self.config.camera_config)
        else:
            self._camera_stabilizer = None
            print("Camera Stabilizer disabled")

        # Initialize tracker
        print("\nInitializing DeepSORT Tracker...")
        self._tracker = ObjectTracker(self.config.tracker_config)

        # Processing statistics
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0

    def _filter_detections(self, detections: List[DetectionResult], frame_height: int = 984) -> List[DetectionResult]:
        """
        Filter detections to remove invalid sizes and shapes.
        """
        # Apply NFL filter
        filtered = self._nfl_filter.filter_detections(detections, frame_height)

        # Merge overlapping detections (O-line handling)
        if self.config.nfl_filter_config.merge_overlaps:
            filtered = self._nfl_filter.merge_overlapping_detections(
                filtered,
                iou_threshold=self.config.nfl_filter_config.merge_iou_threshold)

        return filtered
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return detections

        boxes = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = cv2.dnn.NMSBoxes(boxes.tolist(),
                                   scores.tolist(),
                                   score_threshold = 0.0,
                                   nms_threshold = 0.4)

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    # code to process an individual frame, use later to iterate through all frames
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

        # Apply NMS
        nms_detections = self._apply_nms(filtered_detections)

        # Temporal aggregation
        if self.config.temporal_config.enabled:
            self._temporal.add_frame(frame_id or self._frames_processed, nms_detections)
            aggregated = self._temporal.get_aggregated(frame_id or self._frames_processed)
        else:
            aggregated = nms_detections

        # camera stablization for each detection
        if self._camera_stabilizer is not None and self._camera_stabilizer.is_ready():
            stabilized_detections = self._camera_stabilizer.stabilize_detections(aggregated)
        else:
            stabilized_detections = aggregated

        # DeepSORT tracking
        tracks = self._tracker.update(frame, stabilized_detections, frame_id)

        # Update camera stabilizer with current frame (for next frame)
        if self._camera_stabilizer is not None:
            self._camera_stabilizer.update(frame)

        # Update statistics
        self._total_processing_time += time.time() - start_time
        self._frames_processed += 1

        return tracks
    
    # using process_frame() over an entire video and store data
    def process_video(self, video_path: str, output_video_path: Optional[str] = None,
                      output_json_path: Optional[str] = None, max_frames: Optional[int] = None) -> TrajectoryStore:
        """
        Process an entire video file and return trajectory data.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing: {video_path.name}")

        # Reset state for new video
        self._tracker.reset()
        self._temporal.reset()
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0

        # Determine output paths
        if output_video_path is None and self.config.save_video:
            output_video_path = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
        if output_json_path is None and self.config.save_trajectories:
            output_json_path = str(video_path.parent / f"{video_path.stem}_trajectories.json")

        video_writer = None

        with VideoLoader(str(video_path)) as video:
            total_frames = min(video.metadata.total_frames, max_frames or float('inf'))
            print(f"Video metadata: {video.metadata}")
            print(f"Total frames to process: {int(total_frames)}")

            for frame_id, frame in video:
                if frame_id >= total_frames:
                    break

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
                    print(f"Processed frame {frame_id}/{int(total_frames)}")

        # Finalize video
        if video_writer:
            video_writer.release()
            print(f"Output video saved: {output_video_path}")

        # Save trajectories as JSON
        traj_store = self._tracker.get_trajectory_store()
        if output_json_path:
            self._save_trajectories_json(traj_store, output_json_path)

        return traj_store
    
    def _save_trajectories_json(self, store: TrajectoryStore, filepath: str) -> None:
        """
        Save trajectories to JSON format for future route classification tasks.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {'metadata': {'num_trajectories': store.num_trajectories,
                            'total_detections': store.total_detections,
                            'frames_processed': self._frames_processed,
                            'processing_time_seconds': self._total_processing_time},
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

    def reset(self):
        """Reset the pipeline state for processing a new video."""
        self._tracker.reset()
        self._temporal.reset()
        if self._camera_stabilizer is not None:
            self._camera_stabilizer.reset()
        self._total_processing_time = 0.0
        self._frames_processed = 0
        self._total_detections_raw = 0
        self._total_detections_filtered = 0
        print("Pipeline reset complete.")

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        """
        tracker_stats = self._tracker.get_statistics()

        stats = {'frames_processed': self._frames_processed,
                'total_processing_time': self._total_processing_time,
                'average_fps': (self._frames_processed / self._total_processing_time if self._total_processing_time > 0 else 0),
                'raw_detections': self._total_detections_raw,
                'filtered_detections': self._total_detections_filtered,
                'filter_ratio': (self._total_detections_filtered / self._total_detections_raw if self._total_detections_raw > 0 else 0),
                **tracker_stats}

        return stats
    
    def get_camera_motion_stats(self) -> Optional[Dict]:
        """
        Get camera motion statistics if camera stabilizer is enabled.
        """
        if self._camera_stabilizer is None:
            return None
        return self._camera_stabilizer.get_motion_stats()

# =============================================================================
# TEST / DEMO
# =============================================================================


# need anymore?
# if __name__ == "__main__":
#     sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

#     from nfl_route_tracker.core.config import get_default_pipeline_config

#     # Paths
#     test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
#     video_path = test_folder / "test1.mp4"
#     output_folder = test_folder.parent / "viz_output"
#     output_folder.mkdir(exist_ok=True)

#     if not video_path.exists():
#         print(f"Test video not found: {video_path}")
#         print("Using default config for import test...")

#         # Just test imports
#         config = get_default_pipeline_config()
#         print("\nDefault config loaded successfully:")
#         print(f"  - YOLO model: {config.detector_config.model_name}")
#         print(f"  - Temporal window: {config.temporal_config.window_size}")
#         print(f"  - NFL filter enabled: True")
#     else:
#         # Run full pipeline
#         config = get_default_pipeline_config()
#         config.progress_interval = 50
#         config.save_video = True
#         config.save_trajectories = True

#         print("Running pipeline test...")
#         pipeline = DetectionTracker(config)

#         output_video = output_folder / "test_output.mp4"
#         output_json = output_folder / "test_trajectories.json"

#         store = pipeline.process_video(
#             str(video_path),
#             output_video_path=str(output_video),
#             output_json_path=str(output_json)
#         )

#         print("\nProcessing complete!")
#         print(f"Trajectories found: {store.num_trajectories}")
#         print(f"Total detections: {store.total_detections}")

#         stats = pipeline.get_statistics()
#         print("\nStatistics:")
#         for k, v in stats.items():
#             print(f"  {k}: {v}")
