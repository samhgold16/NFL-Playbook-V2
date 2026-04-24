"""
NFL Route Tracker - ByteTrack Tracker Module
===========================================

ByteTrack implementation using Ultralytics native support.
"""

# Core imports
import numpy as np
import yaml
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

# Ultralytics imports
from ultralytics import YOLO

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_route_tracker.detection.player_detector import DetectionResult
from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore
from nfl_route_tracker.core.config import TrackerConfig


@dataclass
class Track:
    """
    Represents a single tracked object from ByteTrack.
    """
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float
    is_confirmed: bool = True
    time_since_update: int = 0
    hits: int = 1

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of track."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[3]


class ByteTrackTracker:
    """
    Multi-object tracker using ByteTrack algorithm via Ultralytics.
    """

    def __init__(self, config: Optional[TrackerConfig] = None, yolo_model = None):
        """
        Initialize ByteTrack tracker.
        Writes a temporary bytetrack YAML from TrackerConfig values.
        """
        self.config = config or TrackerConfig()
        self.yolo_model = yolo_model

        # State management
        self._trajectory_store = TrajectoryStore()
        self._frame_count = 0
        self._total_tracks_created = 0
        self._active_tracks: Dict[int, int] = {}

        # ByteTrack-specific state
        self._track_history: Dict[int, List[Tuple[float, float]]] = {}
        self._lost_tracks: Dict[int, int] = {}
        self._yaml_path: Optional[str] = None

        # filter predictions/ghost boxes
        self._kalman_predictions: Dict[int, Tuple[float, float, float, float]] = {}

        # Statistics
        self._total_detections = 0
        self._filtered_tracks = 0
        self._ghost_boxes_filtered = 0

        # Generate ByteTrack YAML from config
        self._setup_bytetrack_yaml()

    # generate yaml file
    def _setup_bytetrack_yaml(self) -> None:
        """
        Write a ByteTrack YAML file derived entirely from TrackerConfig.
        """
        yaml_content = self._generate_bytetrack_yaml()

        # Create temporary YAML file
        temp_dir = tempfile.gettempdir()
        self._yaml_path = os.path.join(temp_dir, 'bytetrack_nfl.yaml')

        with open(self._yaml_path, 'w') as f:
            f.write(yaml_content)

        # Also save to project directory for reference
        project_yaml = Path(__file__).parent / 'bytetrack.yaml'
        with open(project_yaml, 'w') as f:
            f.write(yaml_content)

    # function that calls from config.py to create a new yaml file if hyperparameters are changed
    def _generate_bytetrack_yaml(self) -> str:
        """
        Generate ByteTrack YAML content from config parameters.
        """
        # Handle gmc_method (convert None to 'none' for YAML)
        gmc_method = self.config.gmc_method if self.config.gmc_method else 'none'

        return f"""
                # ByteTrack Configuration for NFL All-22 Tracking
                # Modify TrackerConfig in config.py instead

                    # Tracker type
                    tracker_type: bytetrack

                    # Detection confidence thresholds
                    track_high_thresh: {self.config.track_high_thresh}
                    track_low_thresh: {self.config.track_low_thresh}
                    new_track_thresh: {self.config.new_track_thresh}

                    # Occlusion handling
                    track_buffer: {self.config.track_buffer}
                    match_thresh: {self.config.match_thresh}
                    fuse_score: {self.config.fuse_score}

                    # Global Motion Compensation (camera stabilization)
                    # Options: 'ecc' (best), 'sift' (balanced), 'orb' (fastest), 'none' (disabled)
                    gmc_method: {self.config.gmc_method}
                    gmc_downscale: {self.config.gmc_downscale}

                    # ByteTrack specific
                    mot20: False
                    """

    def set_yolo_model(self, model) -> None:
        """
        Set the YOLO model for tracking, called in DetectionTracker
        """
        self.yolo_model = model

    # used when tracking wasnt stable, not needed anymore
    def _is_ghost_box(self, track_id: int, bbox: Tuple[float, float, float, float],
                      confidence: float) -> bool:
        """
        Determine if a track is a ghost box (Kalman prediction without detection).
        """
        if not self.config.filter_ghost_boxes:
            return False

        # If this track was recently lost, it's a ghost prediction
        if track_id in self._lost_tracks:
            return True

        # If the track has very low confidence, it might be a ghost
        if confidence < self.config.track_low_thresh:
            return True

        return False

    def _interpolate_gap(self, track_id: int, gap_size: int,
                         last_bbox: Tuple[float, float, float, float],
                         first_bbox: Tuple[float, float, float, float]) -> List[Detection]:
        """
        Interpolate missing detections for a gap in the trajectory.
        """
        if gap_size <= 0 or gap_size > self.config.max_trajectory_gap:
            return []

        interpolated = []
        last_frame_id = self._lost_tracks.get(track_id, 0) - gap_size

        for i in range(1, gap_size):
            alpha = i / gap_size
            # Linear interpolation
            x = last_bbox[0] + alpha * (first_bbox[0] - last_bbox[0])
            y = last_bbox[1] + alpha * (first_bbox[1] - last_bbox[1])
            w = last_bbox[2] + alpha * (first_bbox[2] - last_bbox[2])
            h = last_bbox[3] + alpha * (first_bbox[3] - last_bbox[3])

            frame_id = last_frame_id + i
            detection = Detection(frame_id = frame_id, x = x, y = y,
                                  width = w, height = h, confidence = 0.0)
            interpolated.append(detection)

        return interpolated

    # main mechanism to process frames and store detections
    def update(self, frame: np.ndarray, detections: List[DetectionResult],
               frame_id: Optional[int] = None) -> List[Track]:
        """
        Update tracker with new detections and return current tracks.
        """
        if frame_id is None:
            frame_id = self._frame_count

        self._frame_count += 1

        if self.yolo_model is None:
            raise RuntimeError("YOLO model not set. Call set_yolo_model() first.")
        
        # Run ByteTrack inference via YOLO
        results = self.yolo_model.track(frame, persist = True,
                                        tracker = self._yaml_path, conf = self.config.confidence_threshold,
                                        iou = self.config.iou_threshold, classes = [0], verbose = False)
        
        tracks = []
        result = results[0]

        seen_track_ids = set()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Extract track info from Ultralytics result
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i
                bbox_xyxy = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())

                seen_track_ids.add(track_id)

                # Check for ghost box
                if self._is_ghost_box(track_id, bbox_xyxy, conf):
                    self._ghost_boxes_filtered += 1
                    continue

                # Store Kalman prediction for this track
                self._kalman_predictions[track_id] = bbox_xyxy

                # Convert xyxy -> (x, y, w, h)
                x1, y1, x2, y2 = bbox_xyxy
                track_bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))

                track = Track(track_id = track_id,
                              bbox = track_bbox,
                              confidence = conf,
                              is_confirmed = True,
                              time_since_update = 0,
                              hits = 1)
                tracks.append(track)

                # Store in trajectory system
                detection = Detection(frame_id = frame_id,
                                      x = track.x,
                                      y = track.y,
                                      width = track.width,
                                      height = track.height,
                                      confidence = track.confidence)
                self._trajectory_store.add_detection(track_id, detection)

                # Track first appearance
                if track_id not in self._active_tracks:
                    self._active_tracks[track_id] = frame_id
                    self._total_tracks_created += 1

                # Rolling position history for motion smoothing (last 30 frames)
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append(track.center)
                if len(self._track_history[track_id]) > 30:
                    self._track_history[track_id].pop(0)

        # Handle lost tracks (gap interpolation)
        for lost_id in list(self._lost_tracks.keys()):
            if lost_id in seen_track_ids:
                # Track was found, interpolate any gap
                if self._lost_tracks[lost_id] > 1 and lost_id in self._kalman_predictions:
                    # interpolate if gap fiund
                    gap_size = self._lost_tracks[lost_id]
                    if gap_size <= self.config.max_trajectory_gap:
                        # Get last known position and interpolate to current
                        last_traj = None
                        for t in self._trajectory_store._trajectories.values():
                            if t.track_id == lost_id:
                                last_traj = t
                                break
                        if last_traj and len(last_traj.detections) > 0:
                            last_det = last_traj.detections[-1]
                            last_bbox = (last_det.x, last_det.y, last_det.width, last_det.height)
                            # Find current bbox
                            for t in tracks:
                                if t.track_id == lost_id:
                                    first_bbox = t.bbox
                                    interpolated = self._interpolate_gap(
                                        lost_id, gap_size, last_bbox, first_bbox
                                    )
                                    for det in interpolated:
                                        self._trajectory_store.add_detection(lost_id, det)
                                    break
                del self._lost_tracks[lost_id]
            else:
                # Track is still lost
                self._lost_tracks[lost_id] = self._lost_tracks.get(lost_id, 0) + 1

        self._total_detections += len(detections)
        self._filtered_tracks += len(tracks)

        return tracks

    def get_trajectory_store(self) -> TrajectoryStore:
        """Get the trajectory store with all tracked data."""
        return self._trajectory_store

    def reset(self) -> None:
        """
        Reset tracker state for new video processing and reset YAML file if needed.
        """
        self._trajectory_store = TrajectoryStore()
        self._frame_count = 0
        self._total_tracks_created = 0
        self._active_tracks = {}
        self._track_history = {}
        self._lost_tracks = {}
        self._total_detections = 0
        self._filtered_tracks = 0

    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {'frames_processed': self._frame_count,
                'total_tracks_created': self._total_tracks_created,
                'total_detections': self._total_detections,
                'filtered_tracks': self._filtered_tracks,
                'avg_detections_per_frame': (self._total_detections / self._frame_count if self._frame_count > 0 else 0),
                'track_completion_rate': (self._filtered_tracks / self._total_detections if self._total_detections > 0 else 0)
        }