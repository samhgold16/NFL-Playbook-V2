"""
NFL Route Tracker - Trajectory Module
=====================================

This module handles the storage and analysis of object trajectories after procesing the videos
"""

# important packages
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path
import pandas as pd


@dataclass
class Detection:
    """
    A single detection of an object in ONE frame.
    """
    frame_id: int
    x: float  
    y: float 
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the detection."""
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        return (center_x, center_y)

    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {'frame_id': self.frame_id,
                'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height,
                'confidence': self.confidence}

    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create Detection from dictionary."""
        clean = {k: v for k, v in data.items()
                if k in ('frame_id', 'x', 'y', 'width', 'height', 'confidence')}
        return cls(**clean)


@dataclass
class Trajectory:
    """
    A sequence of detections for a single tracked object to convert frame analysis to video.
    """
    track_id: int
    detections: List[Detection] = field(default_factory = list)
    metadata: Dict = field(default_factory = dict)

    def add_detection(self, detection: Detection) -> None:
        """
        Add a detection to this trajectory.
        """
        self.detections.append(detection)
        # Keep sorted by frame_id
        self.detections.sort(key = lambda d: d.frame_id)

        # if len(self.detections) % 10 == 0:
        #     print(f"[Trajectory {self.track_id}] Now has {len(self.detections)} detections")

    def get_path(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the path as coordinate arrays.
        """
        if not self.detections:
            return np.array([]), np.array([]), np.array([])

        # pulling attrivutes from dataclass
        frame_ids = np.array([d.frame_id for d in self.detections])
        x_coords = np.array([d.center[0] for d in self.detections])
        y_coords = np.array([d.center[1] for d in self.detections])

        return frame_ids, x_coords, y_coords

    # getter functions for various characterstics to define trajectories
    def get_velocities(self, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous velocities.
        """
        frame_ids, x_coords, y_coords = self.get_path()

        if len(frame_ids) < 2:
            return np.array([]), np.array([])

        # Time difference between frames
        dt = np.diff(frame_ids) / fps 

        # Position differences
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # Velocities
        vx = dx / dt
        vy = dy / dt

        return vx, vy

    # takes from velocity calc and converts to speed
    def get_speeds(self, fps: float = 30.0) -> np.ndarray:
        """
        Compute instantaneous speeds (magnitude of velocity).
        Returns pixels per second.
        """
        vx, vy = self.get_velocities(fps)

        if len(vx) == 0:
            return np.array([])

        return np.sqrt(vx**2 + vy**2)

    def get_total_distance(self) -> float:
        """
        Compute total distance traveled.
        """
        _, x_coords, y_coords = self.get_path()

        if len(x_coords) < 2:
            return 0.0

        # Compute distances between consecutive points
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        distances = np.sqrt(dx**2 + dy**2)

        return float(np.sum(distances))

    # euclidian distance
    def get_displacement(self) -> float:
        """
        Compute displacement (straight-line distance from start to end).
        """
        _, x_coords, y_coords = self.get_path()

        if len(x_coords) < 2:
            return 0.0

        dx = x_coords[-1] - x_coords[0]
        dy = y_coords[-1] - y_coords[0]

        return float(np.sqrt(dx**2 + dy**2))

    def get_frame_range(self) -> Tuple[int, int]:
        """Get the first and last frame IDs."""
        if not self.detections:
            return (0, 0)
        return (self.detections[0].frame_id, self.detections[-1].frame_id)

    def __len__(self) -> int:
        """Number of detections in this trajectory."""
        return len(self.detections)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {'track_id': self.track_id,
                'detections': [d.to_dict() for d in self.detections],
                'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> 'Trajectory':
        """Create Trajectory from dictionary."""
        trajectory = cls(track_id = data['track_id'], metadata = data.get('metadata', {}))
        trajectory.detections = [Detection.from_dict(d) for d in data['detections']]
        return trajectory

    def flip_x_coordinates(self, video_width: float) -> 'Trajectory':
        """
        Flip X coordinates to normalize play direction.
        """
        flipped_traj = Trajectory(track_id=self.track_id, metadata=self.metadata.copy())

        for det in self.detections:
            # Flip the X coordinate (mirrors around vertical center)
            flipped_x = video_width - det.x - det.width
            flipped_det = Detection(frame_id = det.frame_id, x = flipped_x, y = det.y,    
                                    width = det.width, height = det.height, confidence = det.confidence)
            flipped_traj.add_detection(flipped_det)

        return flipped_traj

    def get_field_position(self, video_width: float, video_height: float) -> Dict:
        """
        Get the approximate field position interpretation of this trajectory.
        """
        if len(self.detections) < 2:
            return {'start_x': 0.5, 'start_y': 0.5,
                    'end_x': 0.5, 'end_y': 0.5,
                    'dx': 0, 'dy': 0,
                    'primary_direction': 'unknown'}

        first_det = self.detections[0]
        last_det = self.detections[-1]

        start_x = first_det.center[0] / video_width
        start_y = first_det.center[1] / video_height
        end_x = last_det.center[0] / video_width
        end_y = last_det.center[1] / video_height

        dx = end_x - start_x
        dy = end_y - start_y

        # Primary direction is whichever changed more
        primary_direction = 'vertical' if abs(dx) >= abs(dy) else 'horizontal'

        return {'start_x': start_x, 'start_y': start_y,
                'end_x': end_x, 'end_y': end_y,
                'dx': dx, 'dy': dy,
                'primary_direction': primary_direction}

# takes from Trajectory class
class TrajectoryStore:
    """
    Manages and stores multiple trajectories.
    """
    def __init__(self):
        """Initialize empty trajectory store."""
        # Dictionary mapping track_id -> Trajectory
        self._trajectories: Dict[int, Trajectory] = {}

        # Track some statistics
        self._total_detections: int = 0

    def add_detection(self, track_id: int, detection: Detection) -> None:
        """
        Add a detection to a trajectory.
        """
        # Create trajectory if needed
        if track_id not in self._trajectories:
            self._trajectories[track_id] = Trajectory(track_id=track_id)
            # print(f"[TrajectoryStore] Created new trajectory for track_id={track_id}")

        # Add detection
        self._trajectories[track_id].add_detection(detection)
        self._total_detections += 1

    # can prolly be used later to map track_id to player id/name, when finalized
    def get_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """Get a trajectory by ID, or None if not found."""
        return self._trajectories.get(track_id)

    def get_all_trajectories(self) -> List[Trajectory]:
        """Get all trajectories as a list."""
        return list(self._trajectories.values())

    def get_active_trajectories(self, frame_id: int) -> List[Trajectory]:
        """
        Get trajectories that were active (had detections) at a given frame.
        Useful for: "Which players were being tracked in frame 100?"
        """
        active = []
        for traj in self._trajectories.values():
            start, end = traj.get_frame_range()
            if start <= frame_id <= end:
                active.append(traj)
        return active

    @property
    def num_trajectories(self) -> int:
        """Number of unique trajectories in a video (should be 22 (or less))"""
        return len(self._trajectories)

    @property
    def total_detections(self) -> int:
        """Total number of detections across all trajectories."""
        return self._total_detections

    def to_dataframe(self):
        """
        Convert to pandas DataFrame where row represents one detection per frame
        """

        rows = []
        for traj in self._trajectories.values():
            for det in traj.detections:
                cx, cy = det.center
                rows.append({'track_id': traj.track_id, 'frame_id': det.frame_id, 'x': det.x, 'y': det.y,
                             'width': det.width, 'height': det.height, 'confidence': det.confidence,
                             'center_x': cx, 'center_y': cy})

        return pd.DataFrame(rows)

    def save(self, filepath: str) -> None:
        """
        Save to JSON file.
        """
        filepath = Path(filepath)

        data = {'trajectories': [t.to_dict() for t in self._trajectories.values()],
                'metadata': {'num_trajectories': self.num_trajectories,
                             'total_detections': self.total_detections}}

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryStore':
        """
        Load all trajectories from JSON file.
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        store = cls()
        for traj_data in data['trajectories']:
            traj = Trajectory.from_dict(traj_data)
            store._trajectories[traj.track_id] = traj
            store._total_detections += len(traj.detections)

        print(f"JSON File loaded from {filepath}")
        print(f"{store.num_trajectories} trajectories, "
              f"{store.total_detections} detections")

        return store

    def flip_all_x_coordinates(self, video_width: float) -> 'TrajectoryStore':
        """
        Flip X coordinates for all trajectories in the store.
        """
        flipped_store = TrajectoryStore()

        for traj in self._trajectories.values():
            flipped_traj = traj.flip_x_coordinates(video_width)
            flipped_store._trajectories[flipped_traj.track_id] = flipped_traj
            flipped_store._total_detections += len(flipped_traj)

        return flipped_store

    def get_all_field_positions(self, video_width: float, video_height: float) -> Dict[int, Dict]:
        """
        Get field position analysis for all trajectories.
        """
        positions = {}
        for traj in self._trajectories.values():
            positions[traj.track_id] = traj.get_field_position(video_width, video_height)
        return positions