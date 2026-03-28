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

# importing global variables and other functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.core.video_loader import VideoLoader
from nfl_route_tracker.tracking.trajectory import TrajectoryStore
from nfl_route_tracker.core.config import TrackerConfig, DetectorConfig, DetectionTrackerConfig
from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.tracking.object_tracker import ObjectTracker, Track

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

        # Initialize tracker
        print("\nInitializing DeepSORT Tracker...")
        self._tracker = ObjectTracker(self.config.tracker_config)

        # Processing statistics
        self._total_processing_time = 0.0
        self._frames_processed = 0

    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Filter detections to remove invalid sizes and shapes.

        """
        if not self.config.enable_filtering:
            return detections

        filtered = []

        for det in detections:
            # Calculate metrics
            area = det.width * det.height
            aspect_ratio = det.width / det.height if det.height > 0 else 0

            # Check bounds
            if area < self.config.min_area or area > self.config.max_area:
                continue
            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
                continue

            filtered.append(det)

        return filtered
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        """
        if not self.config.enable_filtering or len(detections) == 0:
            return detections

        # Convert to numpy for NMS
        boxes = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(),
                                   scores.tolist(),
                                   score_threshold = 0.0, 
                                   nms_threshold=self.config.nms_threshold)

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    # code to process an individual frame, use later to iterate through all frames
    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> List[Track]:
        """
        Process a single frame through the detection + tracking pipeline.
        """
        start_time = time.time()

        # detect players
        detections = self._detector.detect(frame)

        # adding filtering logic 
        detections = self._filter_detections(detections)
        detections = self._apply_nms(detections)

        # update tracking based on detections
        tracks = self._tracker.update(frame, detections, frame_id)

        # update statistics
        self._total_processing_time += time.time() - start_time
        self._frames_processed += 1

        return tracks
    
    # using process_frame() over an entire video and store data
    def process_video(self, video_path: str, output_video_path: Optional[str] = None, 
                      max_frames: Optional[int] = None) -> TrajectoryStore:
        """
        Process an entire video file and return trajectory data.
        """

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nNow processing: {video_path.name}")

        # Video writer for output
        video_writer = None

        # Reset tracker state for new video
        self._tracker.reset()
        self._total_processing_time = 0.0
        self._frames_processed = 0

        # Process video
        with VideoLoader(str(video_path)) as video:
            total_frames = video.metadata.total_frames
            # only analyze n number of frames if specified in max_frames = 
            if max_frames:
                total_frames = min(total_frames, max_frames)

            print(f"Total frames to process: {total_frames}")

            for frame_id, frame in video:
                # Check max frames
                if max_frames and frame_id >= max_frames:
                    break

                # Process frame
                tracks = self.process_frame(frame, frame_id)

                # Create output video if requested
                if output_video_path:
                    # Initialize video writer on first frame
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(output_video_path), fourcc,
                                                           video.metadata.fps,
                                                           (w, h))

                    # Draw tracks on frame, defining function below
                    output_frame = self._draw_tracks(frame.copy(), tracks)
                    video_writer.write(output_frame)

                # sanity checking, can delete when not working
                if self.config.verbose and frame_id % self.config.progress_interval == 0:
                    print(f"Processed frame {frame_id}/{total_frames}")

        # Cleanup
        if video_writer:
            video_writer.release()
            print(f"\n Output video saved: {output_video_path}")

        # defining function below for get_trajectory_store(), simply takes from TrajectoryStore
        return self._tracker.get_trajectory_store()
    
    # actual visualization component of stored tracks
    # used in process_video()
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw track bounding boxes and IDs on a frame.
        """

        # Color map for different track IDs
        # Using HSV to get distinct colors
        def get_color(track_id: int) -> Tuple[int, int, int]:
            """Generate a distinct color for each track ID."""
            hue = (int(track_id) * 0.618033988749895) % 1.0
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            return (int(b * 255), int(g * 255), int(r * 255))

        for track in tracks:
            x, y, w, h = track.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            color = get_color(track.track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw track ID label
            label = f"ID: {track.track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Get label size for background
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label background
            cv2.rectangle(frame, (x, y - label_h - 5), (x + label_w, y), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 3), font,
                        font_scale, (255, 255, 255), thickness)

        return frame
    
    # helper, used in process_video()
    def get_trajectory_store(self) -> TrajectoryStore:
        """
        Get the trajectory store with all tracked data.
        """
        return self._tracker.get_trajectory_store()
    
    # helper, used in process_video()
    def reset(self):
        """
        Reset the pipeline state.
        """
        self._tracker.reset()
        self._total_processing_time = 0.0
        self._frames_processed = 0
        print("Reset pipeline complete.")

    # simple getter/setters for overall testing about how peocessing worked
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        """
        tracker_stats = self._tracker.get_statistics()

        return {'frames_processed': self._frames_processed,
                'total_processing_time': self._total_processing_time,
                'average_fps': (self._frames_processed / self._total_processing_time
                                if self._total_processing_time > 0 else 0),
                **tracker_stats}

# file testing! bringing everything together
if __name__ == "__main__":
    # some fix
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

    # Paths (same relative logic as demo_v1.py)
    test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
    video_path = test_folder / "trial_vid2.mp4"
    output_folder = test_folder.parent / "viz_output"
    output_folder.mkdir(exist_ok = True)
    output_video_path = output_folder / "trial_vid_NEW.mp4"
    
    # setting global attributes
    detector_config = DetectorConfig()      
    tracker_config = TrackerConfig()         
    pipeline_config = DetectionTrackerConfig(detector_config = detector_config,
                                             tracker_config = tracker_config)

    # initializing with final class
    pipeline = DetectionTracker(pipeline_config)

    # process video
    store = pipeline.process_video(str(video_path), output_video_path = str(output_video_path))

    print(f"Analyzing complete! Annotated video saved to {output_video_path}")

    # printing results
    stats = pipeline.get_statistics()
    print("\nProcessing statistics:")
    for k, v in stats.items():
        print(f" {k}: {v}")

    print(f"\nTrajectoryStore:")
    print(f"num_trajectories: {store.num_trajectories}")
    print(f"total_detections: {store.total_detections}")