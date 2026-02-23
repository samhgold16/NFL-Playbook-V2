"""
NFL Route Tracker - Trajectory Module
=====================================

This module handles the storage and analysis of object trajectories.
A trajectory is simply a sequence of positions over time.

Author: Sam Gold
Phase: 1 - Foundation

"""

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
    A single detection of an object in one frame, represents what the motion detector found at one moment in time.

    Attributes:
    -----------
    frame_id : Which frame this detection is from (0-indexed)
    x : X coordinate (horizontal position in pixels)
    y : Y coordinate (vertical position in pixels)
    width : Width of the detected region (bounding box)
    height : Height of the detected region (bounding box)
    confidence : How confident we are this is a real object (0.0 to 1.0)

    Properties:
    -----------
    center : Center point of the detection (convenience method)
    area : Area of the bounding box
    """
    frame_id: int
    x: float  # Top-left X
    y: float  # Top-left Y
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
        return {
            'frame_id': self.frame_id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create Detection from dictionary."""
        return cls(**data)


@dataclass
class Trajectory:
    """
    A sequence of detections for a single tracked object.

    Attributes:
    -----------
    track_id :  Unique identifier for this trajectory
    detections : Ordered list of detections (by frame_id)
    metadata :  Additional info (e.g., team, position, player name)

    Key Methods:
    ------------
    add_detection() : Add a new detection to the trajectory
    get_path() : Get just the (x, y) coordinates as arrays
    get_velocities() : Compute instantaneous velocities
    get_total_distance() : Total distance traveled
    """
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_detection(self, detection: Detection) -> None:
        """
        Add a detection to this trajectory.
        """
        self.detections.append(detection)
        # Keep sorted by frame_id
        self.detections.sort(key=lambda d: d.frame_id)

        # if len(self.detections) % 10 == 0:
        #     print(f"[Trajectory {self.track_id}] Now has {len(self.detections)} detections")

    def get_path(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the path as coordinate arrays.

        Returns:
        --------
        Tuple of (frame_ids, x_coords, y_coords)
            - frame_ids: Array of frame numbers
            - x_coords: Array of X positions (center points)
            - y_coords: Array of Y positions (center points)
        """
        if not self.detections:
            return np.array([]), np.array([]), np.array([])

        frame_ids = np.array([d.frame_id for d in self.detections])
        x_coords = np.array([d.center[0] for d in self.detections])
        y_coords = np.array([d.center[1] for d in self.detections])

        return frame_ids, x_coords, y_coords

    def get_velocities(self, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous velocities.

        Parameters:
        -----------
        fps : Frames per second of the source video

        Returns:
        --------
        Tuple of (vx, vy)
            - vx: X velocities (pixels per second)
            - vy: Y velocities (pixels per second)
        """
        frame_ids, x_coords, y_coords = self.get_path()

        if len(frame_ids) < 2:
            return np.array([]), np.array([])

        # Time difference between frames
        dt = np.diff(frame_ids) / fps  # Convert frame diff to seconds

        # Position differences
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # Velocities
        vx = dx / dt
        vy = dy / dt

        return vx, vy

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
        return {
            'track_id': self.track_id,
            'detections': [d.to_dict() for d in self.detections],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Trajectory':
        """Create Trajectory from dictionary."""
        trajectory = cls(
            track_id=data['track_id'],
            metadata=data.get('metadata', {})
        )
        trajectory.detections = [Detection.from_dict(d) for d in data['detections']]
        return trajectory


class TrajectoryStore:
    """
    Manages and stores multiple trajectories.

    Example Usage:
    -------------
    ```python
    store = TrajectoryStore()

    # Add detections
    store.add_detection(track_id=1, detection=Detection(...))
    store.add_detection(track_id=2, detection=Detection(...))

    # Get a specific trajectory
    traj = store.get_trajectory(1)
    print(f"Player 1 traveled {traj.get_total_distance()} pixels")

    # Save results
    store.save("tracking_results.json")

    # Load later
    store = TrajectoryStore.load("tracking_results.json")
    ```
    """

    def __init__(self):
        """Initialize empty trajectory store."""
        # Dictionary mapping track_id -> Trajectory
        self._trajectories: Dict[int, Trajectory] = {}

        # Track some statistics
        self._total_detections: int = 0

        print("[TrajectoryStore] Initialized empty store")

    def add_detection(self, track_id: int, detection: Detection) -> None:
        """
        Add a detection to a trajectory.

        Parameters:
        -----------
        track_id : Which object this detection belongs to
        detection : The detection to add
        """
        # Create trajectory if needed
        if track_id not in self._trajectories:
            self._trajectories[track_id] = Trajectory(track_id=track_id)
            # print(f"[TrajectoryStore] Created new trajectory for track_id={track_id}")

        # Add detection
        self._trajectories[track_id].add_detection(detection)
        self._total_detections += 1

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
        """Number of unique trajectories."""
        return len(self._trajectories)

    @property
    def total_detections(self) -> int:
        """Total number of detections across all trajectories."""
        return self._total_detections

    def get_summary(self) -> str:
        """Get a summary of the stored data."""
        lines = [f"TrajectoryStore Summary:",
                 f"  - Total trajectories: {self.num_trajectories}",
                 f"  - Total detections: {self.total_detections}",
                 f"  - Per-trajectory stats:"
        ]
        for track_id, traj in sorted(self._trajectories.items()):
            start, end = traj.get_frame_range()
            lines.append(
                f"    Track {track_id}: {len(traj)} detections, "
                f"frames {start}-{end}, "
                f"distance: {traj.get_total_distance():.1f}px"
            )
        return "\n".join(lines)

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Each row is one detection.
        Columns: track_id, frame_id, x, y, width, height, confidence, center_x, center_y

        Requires pandas to be installed.
        """

        rows = []
        for traj in self._trajectories.values():
            for det in traj.detections:
                cx, cy = det.center
                rows.append({
                    'track_id': traj.track_id,
                    'frame_id': det.frame_id,
                    'x': det.x,
                    'y': det.y,
                    'width': det.width,
                    'height': det.height,
                    'confidence': det.confidence,
                    'center_x': cx,
                    'center_y': cy
                })

        return pd.DataFrame(rows)

    def save(self, filepath: str) -> None:
        """
        Save to JSON file.

        Parameters:
        -----------
        filepath : str
            Where to save the file
        """
        filepath = Path(filepath)

        data = {
            'trajectories': [t.to_dict() for t in self._trajectories.values()],
            'metadata': {
                'num_trajectories': self.num_trajectories,
                'total_detections': self.total_detections
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[TrajectoryStore] Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryStore':
        """
        Load from JSON file.

        Parameters:
        -----------
        filepath : File to load

        Returns: Loaded store with all trajectories
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        store = cls()
        for traj_data in data['trajectories']:
            traj = Trajectory.from_dict(traj_data)
            store._trajectories[traj.track_id] = traj
            store._total_detections += len(traj.detections)

        print(f"[TrajectoryStore] Loaded from {filepath}")
        print(f"                  {store.num_trajectories} trajectories, "
              f"{store.total_detections} detections")

        return store

# testing, using video from MotionTracker tests to get real detections and trajectories, then test Trajectory methods on them
if __name__ == "__main__":
    from pathlib import Path
    from nfl_route_tracker.tracking.motion_tracker import MotionTracker
    from nfl_route_tracker.core.config import MotionTrackerConfig

    print("\n" + "="*60)
    print("Testing Trajectory Module with Real Video")
    print("="*60 + "\n")

    # Point to your test videos
    test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
    video_path = test_folder / "test_linear.mp4"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
    else:
        # Run motion tracker to get real detections
        config = MotionTrackerConfig(threshold=25, min_contour_area=100, blur_kernel_size=(5, 5))
        tracker = MotionTracker(config)
        store = tracker.process_video(str(video_path))

        print("\n" + "="*60)
        print("TEST 1: TrajectoryStore populated correctly")
        print("="*60)
        assert store.num_trajectories >= 1, "Expected at least 1 trajectory"
        print(store.get_summary())
        print("PASSED!\n")

        # Grab the first trajectory for detailed tests
        traj = store.get_all_trajectories()[0]

        print("="*60)
        print("TEST 2: get_path() returns valid coordinate arrays")
        print("="*60)
        frames, xs, ys = traj.get_path()
        print(f"Frame range: {frames[0]} to {frames[-1]}")
        print(f"X range: {xs.min():.1f} to {xs.max():.1f}")
        print(f"Y range: {ys.min():.1f} to {ys.max():.1f}")
        assert len(frames) == len(xs) == len(ys), "Array lengths should match"
        assert len(frames) > 0, "Should have at least one detection"
        print("PASSED!\n")

        print("="*60)
        print("TEST 3: get_total_distance() vs get_displacement()")
        print("="*60)
        distance = traj.get_total_distance()
        displacement = traj.get_displacement()
        print(f"Total distance traveled: {distance:.1f} pixels")
        print(f"Straight-line displacement: {displacement:.1f} pixels")
        assert distance >= displacement, "Distance should always be >= displacement"
        print("PASSED!\n")

        print("="*60)
        print("TEST 4: get_speeds() returns valid values")
        print("="*60)
        speeds = traj.get_speeds(fps=30.0)
        print(f"Average speed: {speeds.mean():.1f} px/sec")
        print(f"Max speed: {speeds.max():.1f} px/sec")
        assert (speeds >= 0).all(), "Speeds should never be negative"
        print("PASSED!\n")

        print("="*60)
        print("TEST 5: Save and load round-trip")
        print("="*60)
        test_file = Path("test_trajectories_real.json")
        store.save(str(test_file))
        loaded_store = TrajectoryStore.load(str(test_file))
        assert loaded_store.num_trajectories == store.num_trajectories
        assert loaded_store.total_detections == store.total_detections
        test_file.unlink()
        print("PASSED!\n")

        print("="*60)
        print("TEST 6: Convert to DataFrame")
        print("="*60)
        df = store.to_dataframe()
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
        assert len(df) == store.total_detections
        print("PASSED!\n")

        print("="*60)
        print("All Trajectory tests passed with real video data!")
        print("="*60)
