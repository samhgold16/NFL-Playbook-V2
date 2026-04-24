"""NFL Route Tracker - Phase 2: Data Loader
==========================================
Loads and parses trajectory data from Phase 1 JSON output files.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from ..tracking.trajectory import Trajectory, TrajectoryStore, Detection

@dataclass
class ParsedTrajectory:
    """
    A parsed trajectory with additional metadata for Phase 2 processing.
    """
    track_id: int
    detections: List[Detection] = field(default_factory=list)

    # Video metadata
    video_width: float = 0.0
    video_height: float = 0.0

    # Classification labels (to be filled by Phase 2 processors)
    team: Optional[str] = None  # 'offense' or 'defense'
    position: Optional[str] = None  # 'WR', 'TE', 'RB', 'OL', 'DL', 'LB', 'DB', etc.
    is_skill_position: bool = False

    # Metadata from Phase 1
    num_detections: int = 0
    frame_range: Tuple[int, int] = (0, 0)
    total_distance: float = 0.0
    displacement: float = 0.0

    def get_center_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract center coordinates as numpy arrays.
        """
        if not self.detections:
            return np.array([]), np.array([]), np.array([])

        frame_ids = np.array([d.frame_id for d in self.detections])
        x_coords = np.array([d.center[0] for d in self.detections])
        y_coords = np.array([d.center[1] for d in self.detections])

        return frame_ids, x_coords, y_coords

    def get_first_position(self) -> Optional[Tuple[float, float]]:
        """Get the first detection position (x, y) center coordinates."""
        if not self.detections:
            return None
        return self.detections[0].center

    def get_last_position(self) -> Optional[Tuple[float, float]]:
        """Get the last detection position (x, y) center coordinates."""
        if not self.detections:
            return None
        return self.detections[-1].center

    def get_movement_direction(self) -> str:
        """
        Determine primary movement direction.
        """
        _, x_coords, y_coords = self.get_center_coords()

        if len(x_coords) < 2:
            return 'unknown'

        dx = np.abs(x_coords[-1] - x_coords[0])
        dy = np.abs(y_coords[-1] - y_coords[0])

        return 'vertical' if dy >= dx else 'horizontal'

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'track_id': self.track_id,
            'num_detections': self.num_detections,
            'frame_range': self.frame_range,
            'total_distance': self.total_distance,
            'displacement': self.displacement,
            'video_width': self.video_width,
            'video_height': self.video_height,
            'team': self.team,
            'position': self.position,
            'is_skill_position': self.is_skill_position,
            'movement_direction': self.get_movement_direction(),
            'first_position': self.get_first_position(),
            'last_position': self.get_last_position(),
            'detections': [d.to_dict() for d in self.detections]
        }

    @classmethod
    def from_trajectory(cls, traj: Trajectory, video_width: float = 0,
                       video_height: float = 0) -> 'ParsedTrajectory':
        """Create a ParsedTrajectory from a Trajectory object."""
        parsed = cls(
            track_id=traj.track_id,
            detections=traj.detections.copy(),
            video_width=video_width,
            video_height=video_height,
            num_detections=len(traj.detections),
            frame_range=traj.get_frame_range(),
            total_distance=traj.get_total_distance(),
            displacement=traj.get_displacement()
        )
        return parsed


@dataclass
class VideoTrajectories:
    """
    Container for all trajectories from a single video.

    Groups trajectories by video source for easy access and processing.
    """
    video_name: str
    trajectories: List[ParsedTrajectory] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    def get_offense_trajectories(self) -> List[ParsedTrajectory]:
        """Get only offensive player trajectories."""
        return [t for t in self.trajectories if t.team == 'offense']

    def get_defense_trajectories(self) -> List[ParsedTrajectory]:
        """Get only defensive player trajectories."""
        return [t for t in self.trajectories if t.team == 'defense']

    def get_skill_position_trajectories(self) -> List[ParsedTrajectory]:
        """Get only skill position player trajectories (WR, TE, RB)."""
        return [t for t in self.trajectories if t.is_skill_position]

    def add_trajectory(self, traj: ParsedTrajectory) -> None:
        """Add a trajectory to this video."""
        self.trajectories.append(traj)


class TrajectoryDataLoader:
    """
    Data loader for Phase 1 trajectory JSON files.

    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        """
        self.base_path = Path(base_path) if base_path else None

    def load_trajectory_json(self, json_path: Union[str, Path]) -> VideoTrajectories:
        """
        Load a single trajectory JSON file.
        """
        json_path = Path(json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract metadata
        metadata = data.get('metadata', {})

        # Infer video dimensions from the first trajectory if available
        video_width = metadata.get('video_width', 0)
        video_height = metadata.get('video_height', 0)

        # If not in metadata, try to infer from detections
        if not video_width or not video_height:
            for traj_data in data.get('trajectories', []):
                if traj_data.get('detections'):
                    first_det = traj_data['detections'][0]
                    if not video_width:
                        video_width = first_det.get('x', 0) + first_det.get('width', 1920)
                    if not video_height:
                        video_height = first_det.get('y', 0) + first_det.get('height', 984)
                    break

        # Parse trajectories
        trajectories = []
        for traj_data in data.get('trajectories', []):
            # Reconstruct Detection objects
            detections = [Detection.from_dict(d) for d in traj_data.get('detections', [])]

            # Create ParsedTrajectory
            parsed = ParsedTrajectory(
                track_id=traj_data['track_id'],
                detections=detections,
                video_width=video_width,
                video_height=video_height,
                num_detections=traj_data.get('num_detections', len(detections)),
                frame_range=tuple(traj_data.get('frame_range', [0, 0])),
                total_distance=traj_data.get('total_distance', 0.0),
                displacement=traj_data.get('displacement', 0.0)
            )
            trajectories.append(parsed)

        return VideoTrajectories(
            video_name=json_path.stem.replace('_trajectories', ''),
            trajectories=trajectories,
            metadata=metadata
        )

    def load_from_directory(self, directory: Union[str, Path],
                           pattern: str = '*_trajectories.json') -> List[VideoTrajectories]:
        """
        Load all trajectory JSON files from a directory.
        """
        directory = Path(directory)
        json_files = sorted(directory.glob(pattern))

        results = []
        for json_path in json_files:
            try:
                video_traj = self.load_trajectory_json(json_path)
                results.append(video_traj)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")

        return results

    def load_from_viz_output(self, viz_output_dir: Union[str, Path]) -> List[VideoTrajectories]:
        """
        Load all trajectory JSON files from a viz_output directory structure.

        Expected structure:
            viz_output/
            ├── video_name1/
            │   ├── video_name1_tracked.mp4
            │   ├── video_name1_trajectories.json
            │   └── video_name1_trajectories.png
            ├── video_name2/
            │   └── ...
        """
        viz_output_dir = Path(viz_output_dir)
        results = []

        # Iterate through subdirectories
        for subdir in sorted(viz_output_dir.iterdir()):
            if not subdir.is_dir():
                continue

            # Look for trajectory JSON file
            json_files = list(subdir.glob('*_trajectories.json'))

            if json_files:
                try:
                    video_traj = self.load_trajectory_json(json_files[0])
                    results.append(video_traj)
                except Exception as e:
                    print(f"Warning: Failed to load {json_files[0]}: {e}")

        return results

    def get_all_skill_position_trajectories(self, video_trajectories: List[VideoTrajectories]) -> List[ParsedTrajectory]:
        """
        Extract all skill position trajectories from multiple videos.
        """
        all_skill = []
        for video_traj in video_trajectories:
            skill_positions = video_traj.get_skill_position_trajectories()
            all_skill.extend(skill_positions)
        return all_skill

    def get_trajectory_statistics(self,
                                 video_trajectories: List[VideoTrajectories]
                                 ) -> Dict:
        """
        Compute statistics across all loaded videos.
        """
        stats = {
            'num_videos': len(video_trajectories),
            'total_trajectories': 0,
            'total_detections': 0,
            'avg_trajectories_per_video': 0,
            'skill_position_counts': defaultdict(int),
        }

        for video_traj in video_trajectories:
            stats['total_trajectories'] += video_traj.num_trajectories
            stats['total_detections'] += sum(t.num_detections for t in video_traj.trajectories)

            for traj in video_traj.trajectories:
                if traj.is_skill_position:
                    stats['skill_position_counts'][traj.position] += 1

        if stats['num_videos'] > 0:
            stats['avg_trajectories_per_video'] = (
                stats['total_trajectories'] / stats['num_videos']
            )

        return dict(stats)


# Convenience functions

def load_trajectory_from_json(json_path: Union[str, Path]) -> VideoTrajectories:
    """
    Load a single trajectory JSON file.
        VideoTrajectories object
    """
    loader = TrajectoryDataLoader()
    return loader.load_trajectory_json(json_path)


def load_all_trajectories_from_directory(directory: Union[str, Path],
                                        pattern: str = '*_trajectories.json'
                                        ) -> List[VideoTrajectories]:
    """
    Load all trajectory JSON files from a directory.
    """
    loader = TrajectoryDataLoader()
    return loader.load_from_directory(directory, pattern)
