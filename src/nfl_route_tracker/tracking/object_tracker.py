"""
NFL Route Tracker - Object Tracker Module
==========================================

Multi-object tracking using DeepSORT algorithm, combining with YOLO algorithm.
This maintains consistent track IDs across frames even through occlusions.
"""

# important packages
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
#from deep_sort_realtime.deepsort_tracker import DeepSort

# importing global variables and other functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.detection.player_detector import DetectionResult, PlayerDetector
from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore
from nfl_route_tracker.core.config import TrackerConfig, DetectorConfig
from nfl_route_tracker.core.video_loader import VideoLoader

# same set-up as DetectionResults in player_detector
@dataclass
class Track:
    """
    Represents a single tracked object.
    """
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    is_confirmed: bool
    time_since_update: int
    hits: int = 0

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
    
# main class
class ObjectTracker:
    """
    Multi-object tracker using DeepSORT to be used across video frames.
    Should be much more consistent that the nearest-neighbors code from motion_tracker
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize the object tracker using global or specified attributes
        """
        self.config = config or TrackerConfig()

        # importing deepsort
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError("DeepSORT not installed. Install with: pip install deep-sort-realtime")

        # Create DeepSORT tracker using specified configuration
        self.tracker = DeepSort(max_age = self.config.max_age,
                                n_init = self.config.n_init,
                                max_iou_distance = self.config.max_iou_distance,
                                max_cosine_distance = self.config.max_cosine_distance,
                                nn_budget = self.config.nn_budget,
                                embedder = self.config.embedder,
                                embedder_gpu = True)

        # Storage for trajectory data
        self._trajectory_store = TrajectoryStore()

        # Frame counter
        self._frame_count = 0

        # Track statistics
        self._total_tracks_created = 0
        # track_id -> first_seen_frame
        self._active_tracks: Dict[int, int] = {}

    def update(self, frame: np.ndarray, detections: List[DetectionResult], frame_id: Optional[int] = None) -> List[Track]:
        """
        Update tracker with new detections and return current tracks.
        Main method to be called on each iterated frame.
        """
        # Handle frame ID
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count = frame_id + 1

        # Convert detections to DeepSORT format
        # DeepSORT expects: ([left, top, width, height], confidence, class)
        detection_list = []
        for det in detections:
            bbox = [det.x, det.y, det.width, det.height]
            detection_list.append((bbox, det.confidence, det.class_name))

        # Update DeepSORT using deepsort tracking class
        raw_tracks = self.tracker.update_tracks(detection_list, frame=frame)

        # Track hit counts for each track_id
        current_track_hits = {}

        # Convert to Track format and store trajectories from trajectory class
        tracks = []
        for raw_track in raw_tracks:
            # Skip unconfirmed tracks
            if not raw_track.is_confirmed():
                continue

            track_id = raw_track.track_id

            # Get bounding box (ltwh format)
            bbox = raw_track.to_ltwh()

            # Calculate track age (hits) from the trajectory store
            if track_id in self._trajectory_store._trajectories:
                hits = len(self._trajectory_store._trajectories[track_id].detections)
            else:
                hits = 1

            # FILTER: Skip ghost boxes - tracks that haven't been updated
            # These are Kalman filter predictions, not actual detections
            # time_since_update > 0 means no detection was matched to this track
            if self.config.filter_ghost_boxes and raw_track.time_since_update > 1:
                continue

            # FILTER: Require minimum number of hits before showing track
            if hits < self.config.min_hits:
                continue

            # Track first appearance
            if track_id not in self._active_tracks:
                self._active_tracks[track_id] = frame_id
                self._total_tracks_created += 1

            # set all new attributes
            track = Track(track_id = track_id, bbox = (bbox[0], bbox[1], bbox[2], bbox[3]),
                          confidence = raw_track.det_conf if raw_track.det_conf else 0.0,
                          is_confirmed = raw_track.is_confirmed(),
                          time_since_update=raw_track.time_since_update,
                          hits=hits)

            tracks.append(track)

            # Store attributes from above into trajectory class set up from trajectory.py
            detection = Detection(frame_id = frame_id, x = track.x, y = track.y,
                width = track.width, height = track.height, confidence = track.confidence)

            self._trajectory_store.add_detection(track_id, detection)

        return tracks
    
    # getter function that stores trajectories
    def get_trajectory_store(self) -> TrajectoryStore:
        """
        Get the trajectory store with all tracked data.
        """
        return self._trajectory_store
    
    def reset(self):
        """
        Reset the tracker state to be called between videos
        """
        print("Resetting tracker state...")

        # Re-create DeepSORT tracker
        from deep_sort_realtime.deepsort_tracker import DeepSort

        # Create DeepSORT tracker using specified configuration
        self.tracker = DeepSort(max_age = self.config.max_age,
                        n_init = self.config.n_init,
                        max_iou_distance = self.config.max_iou_distance,
                        max_cosine_distance = self.config.max_cosine_distance,
                        nn_budget = self.config.nn_budget,
                        embedder = self.config.embedder,
                        embedder_gpu = True)

        # Reset storage
        self._trajectory_store = TrajectoryStore()
        self._frame_count = 0
        self._total_tracks_created = 0
        self._active_tracks = {}

    # similar summary function to the one in player_detector
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {'frames_processed': self._frame_count,
                'total_tracks_created': self._total_tracks_created,
                'total_detections': self._trajectory_store.total_detections,
                'avg_detections_per_frame': (
                    self._trajectory_store.total_detections / self._frame_count
                    if self._frame_count > 0 else 0)}
    

# probably can delete now???
    
# file testing! similar process to player_detector.py
# if __name__ == "__main__":
#     # some fix
#     sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

#     # Paths (same relative logic as demo_v1.py)
#     test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
#     video_path = test_folder / "trial_vid.mp4"
#     output_folder = test_folder.parent / "viz_output"
#     output_folder.mkdir(exist_ok = True)

#     # loading video and getting just one frame to analyze simple detection
#     with VideoLoader(video_path) as loader:
#         print(loader.metadata)          
#         frames = [loader.get_frame(50), loader.get_frame(51), loader.get_frame(52)]

#     # using config settings, first tracking players
#     detector_config = DetectorConfig()
#     detector = PlayerDetector(detector_config)

#     # now testing the actual file -> tracking objects
#     tracker_config = TrackerConfig()
#     tracker = ObjectTracker(tracker_config)

#     # detect and track for set of frames
#     all_tracks = []
#     for i, frame in enumerate(frames):
#         print(f"\n--- Frame {50+i} ---")
#         detections = detector.detect(frame, verbose=True)
#         tracks = tracker.update(frame, detections, frame_id = 50 + i)
#         print(f"Confirmed tracks: {len(tracks)}")
#         all_tracks.extend(tracks)
    
#     print(f"\nFINAL RESULT: {len(set(t.track_id for t in all_tracks))} unique track IDs")
#     # show first 5 tracks as example
#     for track in all_tracks[:5]: 
#         print(f"Track ID {track.track_id}: center=({track.center[0]:.1f}, {track.center[1]:.1f})")