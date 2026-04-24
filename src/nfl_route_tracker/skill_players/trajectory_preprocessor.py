"""
NFL Route Tracker - Phase 2: Trajectory Preprocessor
======================================================
Normalizes extracted trajectories to match synthetic route format for training.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# =============================================================================
# Files to exclude from processing (combined files, not actual videos)
# =============================================================================

EXCLUDED_FILE_PATTERNS = [
    'all_skill_positions',
    'extraction_statistics',
    'all_normalized',
]

# =============================================================================
# Normalization Config
# =============================================================================

@dataclass
class NormalizationConfig:
    """Configuration for trajectory normalization."""
    # Output settings
    output_dir: Path = Path("data/normalized_trajectories")

    # Trajectory settings
    num_points: int = 64          # Fixed number of points for all trajectories
    normalize_range: Tuple[float, float] = (0.0, 1.0)  # Normalize to [0, 1]

    # Field orientation settings
    flip_x: bool = True           # Flip X to match synthetic route orientation
    flip_y: bool = False           # Flip Y if needed
    invert_y: bool = True          # Invert Y (camera perspective: Y increases downward)

    # Filtering
    min_points: int = 10          # Minimum detections to consider valid
    max_points: int = 500         # Maximum detections to truncate

    # Class labels (for manual labeling integration)
    route_classes: List[str] = None

    def __post_init__(self):
        if self.route_classes is None:
            self.route_classes = ['streak', 'slant', 'post', 'corner', 'drag', 'curl', 'dig', 'out', 'comeback', 'flat', 'wheel', 'unknown']

# =============================================================================
# Normalized Trajectory Dataclass
# =============================================================================

@dataclass
class NormalizedTrajectory:
    """
    A trajectory normalized to match synthetic route format.
    """
    # Identification
    track_id: int
    source_video: str

    # Normalized coordinates (64 points, [0,1] range)
    x_coords: np.ndarray
    y_coords: np.ndarray
    num_points: int

    # Metadata
    original_num_detections: int = 0
    first_position: Tuple[float, float] = (0.0, 0.0)  # Normalized
    last_position: Tuple[float, float] = (0.0, 0.0)   # Normalized

    # Classification (filled by manual labeling or model inference)
    route_label: Optional[str] = 'unknown'
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': self.track_id,
            'source_video': self.source_video,
            'x_coords': self.x_coords.tolist(),
            'y_coords': self.y_coords.tolist(),
            'num_points': self.num_points,
            'original_num_detections': self.original_num_detections,
            'first_position': self.first_position,
            'last_position': self.last_position,
            'route_label': self.route_label,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NormalizedTrajectory':
        """Create from dictionary."""
        return cls(
            track_id=data['track_id'],
            source_video=data['source_video'],
            x_coords=np.array(data['x_coords']),
            y_coords=np.array(data['y_coords']),
            num_points=data['num_points'],
            original_num_detections=data.get('original_num_detections', 0),
            first_position=tuple(data.get('first_position', (0, 0))),
            last_position=tuple(data.get('last_position', (0, 0))),
            route_label=data.get('route_label', 'unknown'),
            confidence=data.get('confidence', 0.0),
        )

    def to_training_format(self) -> Dict:
        """Convert to format for CNN training."""
        return {
            'x_coords': self.x_coords.tolist(),
            'y_coords': self.y_coords.tolist(),
            'route_type': self.route_label,
        }


# =============================================================================
# Trajectory Normalizer
# =============================================================================

class TrajectoryNormalizer:
    """
    Normalizes extracted trajectories to match synthetic route format.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()

    def _extract_center_from_detection(self, det: Dict) -> Tuple[float, float]:
        """
        Extract center coordinates from a detection.
        """
        x = det.get('x', 0)
        y = det.get('y', 0)
        width = det.get('width', 0)
        height = det.get('height', 0)

        # Calculate center of bounding box
        center_x = x + width / 2.0
        center_y = y + height / 2.0

        return center_x, center_y

    def normalize_trajectory(self, trajectory_data: Dict, video_width: float, video_height: float) -> Optional[NormalizedTrajectory]:
        """
        Normalize a single trajectory from phase2_output.
        """
        # Extract detection centers
        detections = trajectory_data.get('detections', [])
        if len(detections) < self.config.min_points:
            return None

        # Extract center coordinates using the detection format
        x_coords_raw = []
        y_coords_raw = []
        for det in detections:
            center_x, center_y = self._extract_center_from_detection(det)
            x_coords_raw.append(center_x)
            y_coords_raw.append(center_y)

        x_raw = np.array(x_coords_raw)
        y_raw = np.array(y_coords_raw)

        # Normalize to [0, 1]
        # Use video dimensions from metadata for proper normalization
        if video_width > 0:
            x_norm = x_raw / video_width
        else:
            x_norm = x_raw / 1920.0  # Fallback

        if video_height > 0:
            if self.config.invert_y:
                # Invert Y (in video, Y increases downward, field view inverts this)
                y_norm = 1.0 - (y_raw / video_height)
            else:
                y_norm = y_raw / video_height
        else:
            y_norm = y_raw / 1080.0  # Fallback

        # Clip to [0, 1]
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Resample to fixed number of points
        x_resampled, y_resampled = self._resample_trajectory(x_norm, y_norm, self.config.num_points)

        # Store first and last positions
        first_pos = (float(x_resampled[0]), float(y_resampled[0]))
        last_pos = (float(x_resampled[-1]), float(y_resampled[-1]))

        return NormalizedTrajectory(track_id=trajectory_data.get('track_id', 0),
                                    source_video=trajectory_data.get('source_video', 'unknown'),
                                    x_coords=x_resampled,
                                    y_coords=y_resampled,
                                    num_points=len(x_resampled),
                                    original_num_detections=len(detections),
                                    first_position=first_pos,
                                    last_position=last_pos,
                                    route_label='unknown',  # Will be filled by manual labeling or inference
                                    confidence=0.0)

    def _resample_trajectory(self, x: np.ndarray, y: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample trajectory to fixed number of points using linear interpolation.
        """
        if len(x) == num_points:
            return x, y

        # Arc-length parameterization
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])

        if cumulative_dist[-1] == 0:
            # All points are the same - return constant array
            return np.full(num_points, x[0]), np.full(num_points, y[0])

        # Normalize to [0, 1]
        cumulative_dist = cumulative_dist / cumulative_dist[-1]

        # Create output parameter values
        t_out = np.linspace(0, 1, num_points)

        # Interpolate
        x_resampled = np.interp(t_out, cumulative_dist, x)
        y_resampled = np.interp(t_out, cumulative_dist, y)

        return x_resampled, y_resampled

    def normalize_from_json(self, json_path: Path) -> Tuple[List[NormalizedTrajectory], bool]:
        """
        Load and normalize all trajectories from a phase2_output JSON file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        video_name = data.get('video_name', json_path.stem)
        trajectories = data.get('trajectories', [])

        # Get video dimensions from metadata (these are in the skill_positions.json)
        video_width = data.get('video_width', 1920.0)
        video_height = data.get('video_height', 1080.0)

        # If still not set, try to get from trajectory metadata
        if video_width <= 0 or video_height <= 0:
            for traj in trajectories:
                if 'video_width' in traj and traj['video_width'] > 0:
                    video_width = traj['video_width']
                    video_height = traj['video_height']
                    break

        # Fallback to defaults if still not set
        if video_width <= 0:
            video_width = 1920.0
        if video_height <= 0:
            video_height = 1080.0

        normalized = []
        for traj_data in trajectories:
            traj_data['source_video'] = video_name
            normalized_traj = self.normalize_trajectory(traj_data, video_width, video_height)
            if normalized_traj is not None:
                normalized.append(normalized_traj)

        return normalized, False


# =============================================================================
# Batch Processing
# =============================================================================

class TrajectoryPreprocessor:
    """
    Batch processor for normalizing multiple trajectory files.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.normalizer = TrajectoryNormalizer(config)

    def _should_exclude_file(self, filename: str) -> bool:
        """Check if a file should be excluded from processing."""
        filename_lower = filename.lower()
        for pattern in EXCLUDED_FILE_PATTERNS:
            if pattern.lower() in filename_lower:
                return True
        return False

    def run(self, input_dir: Path, output_dir: Path) -> Dict:
        """
        Process all trajectory files in input directory.
        """
        print("=" * 60)
        print("NFL Route Tracker - Trajectory Preprocessor")
        print("=" * 60)
        print(f"\nInput directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Num points:       {self.config.num_points}")
        print(f"Min detections:   {self.config.min_points}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all skill position JSON files
        json_files = list(input_dir.glob("*_skill_positions.json"))

        if not json_files:
            print("ERROR: No *_skill_positions.json files found!")
            return {'total_files': 0, 'total_trajectories': 0}

        print(f"\nFound {len(json_files)} file(s) to process...\n")

        # Filter out excluded files
        valid_files = [f for f in json_files if not self._should_exclude_file(f.stem)]
        excluded_count = len(json_files) - len(valid_files)

        if excluded_count > 0:
            print(f"Excluding {excluded_count} combined/non-video file(s)")
            for f in json_files:
                if self._should_exclude_file(f.stem):
                    print(f"  - Skipped: {f.stem}")
            print()

        all_normalized = []
        stats = {
            'total_files': len(valid_files),
            'total_trajectories': 0,
            'videos_processed': [],
            'excluded_files': excluded_count,
        }

        for json_path in sorted(valid_files):
            video_name = json_path.stem.replace('_skill_positions', '')
            print(f"Processing: {video_name}")

            try:
                normalized, skipped = self.normalizer.normalize_from_json(json_path)

                # Save individual video result
                video_output = output_dir / f"{video_name}_normalized.json"
                self._save_normalized(normalized, video_output)

                print(f"  {len(normalized)} trajectories normalized")

                stats['total_trajectories'] += len(normalized)
                stats['videos_processed'].append(video_name)
                all_normalized.extend(normalized)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Save combined dataset
        if all_normalized:
            combined_path = output_dir / "all_normalized_trajectories.json"
            self._save_combined(all_normalized, combined_path)
            print(f"\nSaved combined dataset: {combined_path}")

        self._print_summary(stats)

        return stats

    def _save_normalized(self, trajectories: List[NormalizedTrajectory], output_path: Path):
        """Save normalized trajectories for a single video."""
        data = {
            'num_trajectories': len(trajectories),
            'trajectories': [t.to_dict() for t in trajectories]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_combined(self, trajectories: List[NormalizedTrajectory], output_path: Path):
        """Save all normalized trajectories in one file."""
        data = {
            'total_count': len(trajectories),
            'num_points': self.config.num_points,
            'trajectories': [t.to_dict() for t in trajectories]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _print_summary(self, stats: Dict):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed:      {stats['total_files']}")
        print(f"Files excluded:      {stats.get('excluded_files', 0)}")
        print(f"Trajectories saved: {stats['total_trajectories']}")
        print("=" * 60)


# =============================================================================
# Integration with Training Data Prep
# =============================================================================

def load_normalized_for_training(normalized_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load normalized trajectories for training.
    """
    combined_path = normalized_dir / "all_normalized_trajectories.json"

    if not combined_path.exists():
        print(f"Warning: {combined_path} not found")
        return np.array([]), []

    with open(combined_path, 'r') as f:
        data = json.load(f)

    trajectories = data.get('trajectories', [])
    num_points = data.get('num_points', 64)

    X = np.zeros((len(trajectories), num_points, 2), dtype=np.float32)
    metadata = []

    for i, traj in enumerate(trajectories):
        x_coords = np.array(traj['x_coords'])
        y_coords = np.array(traj['y_coords'])

        # Pad or truncate to num_points
        if len(x_coords) < num_points:
            x_coords = np.pad(x_coords, (0, num_points - len(x_coords)), mode='edge')
            y_coords = np.pad(y_coords, (0, num_points - len(y_coords)), mode='edge')
        else:
            x_coords = x_coords[:num_points]
            y_coords = y_coords[:num_points]

        X[i, :, 0] = x_coords
        X[i, :, 1] = y_coords

        metadata.append({'track_id': traj['track_id'],
                        'source_video': traj['source_video'],
                        'route_label': traj.get('route_label', 'unknown'),
                        'confidence': traj.get('confidence', 0.0),})

    return X, metadata


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Normalize extracted trajectories for training", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input-dir", type=Path, default=Path("data/phase2_output"), help="Directory with *_skill_positions.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/normalized_trajectories"), help="Output directory for normalized trajectories")
    parser.add_argument("--num-points", type=int, default=64, help="Number of points in output trajectories (default: 64)")
    parser.add_argument("--min-detections", type=int, default=10, help="Minimum detections for valid trajectory (default: 10)")

    args = parser.parse_args()

    config = NormalizationConfig(output_dir=args.output_dir,
                                 num_points=args.num_points,
                                 min_points=args.min_detections,)

    processor = TrajectoryPreprocessor(config)
    processor.run(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()