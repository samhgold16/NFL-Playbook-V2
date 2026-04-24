"""
NFL Route Tracker - Phase 2: Skill Position Limiter
====================================================
Post-processing filter to limit skill positions per play based on displacement.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import shutil


# =============================================================================
# Configuration
# =============================================================================

# Files to exclude from processing - these are combined/aggregate files
EXCLUDED_FILE_PATTERNS = [
    'all_skill_positions',      # Combined file with all videos
    'extraction_statistics',   # Statistics summary file
    'all_normalized',          # Combined normalized file
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SkillPositionLimiterConfig:
    """Configuration for skill position limiting."""
    # Filtering parameters
    max_skill_positions: int = 8       # Maximum skill positions to keep per video
    min_displacement: float = 50.0    # Minimum displacement (pixels) to consider

    # Filtering behavior
    only_positive_displacement: bool = True  # Only keep if moved in expected direction
    reclassify_removed: bool = True    # Mark excluded as non-skill position


# =============================================================================
# Displacement Limiter
# =============================================================================

class SkillPositionLimiter:
    """
    Post-filter for skill positions to limit candidates per play.

    This filter:
    1. Loads skill positions from phase2_output JSON files
    2. For each video, ranks trajectories by displacement
    3. Keeps only top N by displacement (configurable)
    4. Marks excluded trajectories as non-skill position
    5. Saves filtered results

    DISPLACEMENT vs TOTAL_DISTANCE:
    ==============================
    - displacement: Straight-line distance from first to last detection center
                     (measures how far player moved from start to end)
    - total_distance: Sum of all frame-to-frame movements
                      (measures total path length)

    For route classification, displacement is more useful because:
    - Route receivers typically run routes of significant length
    - Linemen or players in the backfield may have high total_distance
      but low displacement (blocking assignments)
    """

    def __init__(self, config: Optional[SkillPositionLimiterConfig] = None):
        self.config = config or SkillPositionLimiterConfig()

    def _is_excluded_file(self, filename: str) -> bool:
        """Check if file should be excluded from processing."""
        return any(pattern in filename for pattern in EXCLUDED_FILE_PATTERNS)

    def _calculate_displacement_from_detections(
        self,
        detections: List[Dict]
    ) -> Tuple[float, bool]:
        """
        Calculate displacement from detections.

        Returns:
            Tuple of (displacement, is_positive_direction)
            - displacement: Euclidean distance from first to last detection center
            - is_positive_direction: True if last center is "downfield" relative to first
        """
        if len(detections) < 2:
            return 0.0, False

        # Get first detection center
        first = detections[0]
        first_x = first.get('x', 0) + first.get('width', 0) / 2
        first_y = first.get('y', 0) + first.get('height', 0) / 2

        # Get last detection center
        last = detections[-1]
        last_x = last.get('x', 0) + last.get('width', 0) / 2
        last_y = last.get('y', 0) + last.get('height', 0) / 2

        # Calculate Euclidean distance
        dx = last_x - first_x
        dy = last_y - first_y
        displacement = (dx**2 + dy**2)**0.5

        # In video coordinates, Y increases downward
        # "Positive" direction depends on camera angle, but typically
        # we want players who moved a meaningful distance
        # The displacement value itself is the primary filter
        is_positive = displacement > 0

        return displacement, is_positive

    def _rank_by_displacement(
        self,
        trajectories: List[Dict]
    ) -> List[Tuple[int, float, Dict]]:
        """
        Rank trajectories by displacement.

        Returns:
            List of (index, displacement, trajectory) sorted by displacement descending
        """
        ranked = []
        for i, traj in enumerate(trajectories):
            # Use stored displacement if available
            displacement = traj.get('displacement', 0.0)

            # Recalculate from detections if not stored
            if displacement == 0.0 and traj.get('detections'):
                displacement, _ = self._calculate_displacement_from_detections(traj['detections'])

            ranked.append((i, displacement, traj))

        # Sort by displacement descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def filter_trajectories(
        self,
        trajectories: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Filter trajectories to max skill positions per video.

        Args:
            trajectories: List of trajectory dicts from phase2_output

        Returns:
            Tuple of (filtered_trajectories, statistics)
        """
        # Separate skill and non-skill positions
        skill_positions = []
        non_skill_positions = []

        for traj in trajectories:
            if traj.get('is_skill_position', False):
                skill_positions.append(traj)
            else:
                non_skill_positions.append(traj)

        original_skill_count = len(skill_positions)
        excluded_by_count = 0
        excluded_by_displacement = 0

        # Apply minimum displacement filter first
        filtered_skill = []
        for traj in skill_positions:
            displacement = traj.get('displacement', 0.0)

            # Recalculate if needed
            if displacement == 0.0 and traj.get('detections'):
                displacement, _ = self._calculate_displacement_from_detections(traj['detections'])

            if displacement < self.config.min_displacement:
                excluded_by_displacement += 1
                if self.config.reclassify_removed:
                    traj_copy = traj.copy()
                    traj_copy['is_skill_position'] = False
                    traj_copy['exclusion_reason'] = 'below_min_displacement'
                    traj_copy['exclusion_displacement'] = displacement
                    non_skill_positions.append(traj_copy)
                continue

            filtered_skill.append(traj)

        # Rank by displacement and keep top N
        if len(filtered_skill) > self.config.max_skill_positions:
            ranked = self._rank_by_displacement(filtered_skill)
            kept = ranked[:self.config.max_skill_positions]
            excluded = ranked[self.config.max_skill_positions:]

            excluded_by_count = len(excluded)

            # Reclassify excluded trajectories
            if self.config.reclassify_removed:
                for _, disp, traj in excluded:
                    traj_copy = traj.copy()
                    traj_copy['is_skill_position'] = False
                    traj_copy['exclusion_reason'] = 'exceeded_max_positions'
                    traj_copy['exclusion_rank'] = self.config.max_skill_positions + 1
                    traj_copy['exclusion_displacement'] = disp
                    non_skill_positions.append(traj_copy)

            filtered_skill = [traj for _, _, traj in kept]

        # Combine all trajectories
        all_trajectories = filtered_skill + non_skill_positions

        # Build statistics
        stats = {
            'original_skill_count': original_skill_count,
            'excluded_by_displacement': excluded_by_displacement,
            'excluded_by_count': excluded_by_count,
            'final_skill_count': len(filtered_skill),
            'non_skill_count': len(non_skill_positions),
            'total_trajectories': len(all_trajectories),
        }

        return all_trajectories, stats

    def process_file(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Process a single skill_positions.json file.

        Args:
            input_path: Path to input *_skill_positions.json
            output_path: Path to save filtered output

        Returns:
            Processing statistics
        """
        # Load data
        with open(input_path, 'r') as f:
            data = json.load(f)

        video_name = data.get('video_name', input_path.stem.replace('_skill_positions', ''))
        trajectories = data.get('trajectories', [])

        # Filter trajectories
        filtered_trajectories, stats = self.filter_trajectories(trajectories)

        # Create output
        output_data = data.copy()
        output_data['trajectories'] = filtered_trajectories
        output_data['limiter_applied'] = True
        output_data['limiter_stats'] = stats
        output_data['limiter_config'] = {
            'max_skill_positions': self.config.max_skill_positions,
            'min_displacement': self.config.min_displacement,
            'only_positive_displacement': self.config.only_positive_displacement,
        }

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        return stats

    def run(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Process all skill_positions.json files in directory.

        Args:
            input_dir: Directory containing *_skill_positions.json files
            output_dir: Output directory (if None, creates *_limited.json files)

        Returns:
            Overall processing statistics
        """
        print("=" * 60)
        print("NFL Route Tracker - Skill Position Limiter")
        print("=" * 60)
        print(f"\nInput directory:  {input_dir}")
        print(f"Output directory: {output_dir or 'auto (same dir with _limited suffix)'}")
        print(f"Max positions:    {self.config.max_skill_positions}")
        print(f"Min displacement: {self.config.min_displacement}")

        # Default output to same dir with _limited suffix
        if output_dir is None:
            output_dir = input_dir

        output_dir = Path(output_dir)

        # Find all skill position files
        skill_files = list(input_dir.glob("*_skill_positions.json"))

        # Filter out excluded files (combined/aggregate files)
        skill_files = [f for f in skill_files if not self._is_excluded_file(f.name)]

        if not skill_files:
            print("ERROR: No individual *_skill_positions.json files found!")
            print(f"      (Excluded combined files like 'all_skill_positions.json')")
            return {'total_files': 0, 'total_excluded': 0}

        print(f"\nFound {len(skill_files)} file(s) to process...\n")

        # Process each file
        total_stats = {
            'total_files': len(skill_files),
            'total_excluded': 0,
            'files_with_filtering': 0,
            'videos': {},
        }

        for json_path in sorted(skill_files):
            video_name = json_path.stem.replace('_skill_positions', '')

            # Determine output path
            if output_dir == input_dir:
                # Create in same dir with _limited suffix
                output_path = json_path.parent / f"{video_name}_limited_skill_positions.json"
            else:
                output_path = output_dir / json_path.name

            print(f"Processing: {video_name}")

            try:
                stats = self.process_file(json_path, output_path)
                total_stats['total_excluded'] += (
                    stats['excluded_by_count'] + stats['excluded_by_displacement']
                )
                total_stats['videos'][video_name] = stats

                if stats['excluded_by_count'] > 0 or stats['excluded_by_displacement'] > 0:
                    total_stats['files_with_filtering'] += 1

                excluded_total = stats['excluded_by_count'] + stats['excluded_by_displacement']
                if excluded_total > 0:
                    print(f"  -> Skill positions: {stats['original_skill_count']} -> "
                          f"{stats['final_skill_count']} "
                          f"(excluded: {excluded_total})")
                else:
                    print(f"  -> Skill positions: {stats['final_skill_count']} (no filtering needed)")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        self._print_summary(total_stats)

        return total_stats

    def _print_summary(self, stats: Dict):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("LIMITING SUMMARY")
        print("=" * 60)
        print(f"Files processed:        {stats['total_files']}")
        print(f"Files with filtering:   {stats['files_with_filtering']}")
        print(f"Total excluded:         {stats['total_excluded']}")

        # Print videos that had filtering
        if stats['videos']:
            filtering_occurred = False
            for video, v_stats in stats['videos'].items():
                if v_stats['excluded_by_count'] > 0 or v_stats['excluded_by_displacement'] > 0:
                    if not filtering_occurred:
                        print("\nVideos with changes:")
                        filtering_occurred = True
                    excluded = v_stats['excluded_by_count'] + v_stats['excluded_by_displacement']
                    print(f"  {video}: {v_stats['original_skill_count']} -> "
                          f"{v_stats['final_skill_count']} (-{excluded})")

        print("=" * 60)


# =============================================================================
# Pipeline Integration Helper
# =============================================================================

def limit_skill_positions(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    max_positions: int = 8,
    min_displacement: float = 50.0
) -> Dict:
    """
    Convenience function to run skill position limiting.

    This can be called after skill_position_filter.py in your pipeline.

    Args:
        input_dir: Directory with phase2_output files
        output_dir: Optional output directory (None = same dir with _limited suffix)
        max_positions: Maximum skill positions per play
        min_displacement: Minimum displacement threshold in pixels

    Returns:
        Processing statistics
    """
    config = SkillPositionLimiterConfig(
        max_skill_positions=max_positions,
        min_displacement=min_displacement,
    )

    limiter = SkillPositionLimiter(config)
    return limiter.run(input_dir, output_dir)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Limit skill positions to max per play based on displacement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - limit to 8 skill positions per play
  python -m nfl_route_tracker.phase2.skill_position_limiter \\
      --input-dir data/phase2_output \\
      --output-dir data/phase2_output_limited

  # More aggressive filtering - 5 max, 100px minimum displacement
  python -m nfl_route_tracker.phase2.skill_position_limiter \\
      --input-dir data/phase2_output \\
      --max-positions 5 \\
      --min-displacement 100.0

  # Create limited files in same directory
  python -m nfl_route_tracker.phase2.skill_position_limiter \\
      --input-dir data/phase2_output

  # View what would be filtered without modifying
  python -m nfl_route_tracker.phase2.skill_position_limiter \\
      --input-dir data/phase2_output \\
      --dry-run

The limiter will:
1. Keep only the top N skill positions by displacement
2. Exclude positions below minimum displacement threshold
3. Mark excluded trajectories as non-skill position
        """
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/phase2_output"),
        help="Directory with *_skill_positions.json files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase2_output_cleaned"),
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=7,
        help="Maximum skill positions to keep per video (default: 7)"
    )
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=50.0,
        help="Minimum displacement threshold in pixels (default: 50.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filtered without modifying files"
    )

    args = parser.parse_args()

    config = SkillPositionLimiterConfig(
        max_skill_positions=args.max_positions,
        min_displacement=args.min_displacement,
    )

    limiter = SkillPositionLimiter(config)
    input_dir = Path(args.input_dir)

    if args.dry_run:
        # Just show stats without modifying
        print("DRY RUN - No files will be modified\n")

        skill_files = [f for f in input_dir.glob("*_skill_positions.json")
                       if not limiter._is_excluded_file(f.name)]
        if not skill_files:
            print("No skill_positions files found (excluded combined files).")
            return

        for json_path in sorted(skill_files):
            with open(json_path, 'r') as f:
                data = json.load(f)

            trajectories = data.get('trajectories', [])
            skill_positions = [t for t in trajectories if t.get('is_skill_position', False)]
            skill_count = len(skill_positions)

            # Calculate displacements
            displacements = []
            for traj in skill_positions:
                disp = traj.get('displacement', 0.0)
                if disp == 0.0 and traj.get('detections'):
                    detections = traj['detections']
                    if len(detections) >= 2:
                        first = detections[0]
                        last = detections[-1]
                        fx = first.get('x', 0) + first.get('width', 0) / 2
                        fy = first.get('y', 0) + first.get('height', 0) / 2
                        lx = last.get('x', 0) + last.get('width', 0) / 2
                        ly = last.get('y', 0) + last.get('height', 0) / 2
                        disp = ((lx - fx)**2 + (ly - fy)**2)**0.5
                displacements.append(disp)

            # Sort by displacement
            sorted_disp = sorted(displacements, reverse=True)

            video_name = json_path.stem.replace('_skill_positions', '')
            if skill_count > config.max_skill_positions:
                top_8_disp = sorted_disp[config.max_skill_positions - 1] if len(sorted_disp) >= config.max_skill_positions else 0
                below_thresh = sum(1 for d in sorted_disp if d < config.min_displacement)
                print(f"{video_name}:")
                print(f"  Skill positions: {skill_count}")
                print(f"  Would keep: {config.max_skill_positions} (top by displacement)")
                print(f"  Below min displacement ({config.min_displacement}): {below_thresh}")
                print(f"  Threshold displacement: {top_8_disp:.1f}px")
                print()
            else:
                print(f"{video_name}: {skill_count} skill positions (no change needed)\n")
    else:
        limiter.run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()