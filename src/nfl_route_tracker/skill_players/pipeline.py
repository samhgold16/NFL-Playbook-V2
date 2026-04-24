"""NFL Route Tracker - Phase 2: Complete Pipeline
=================================================
Batch processing pipeline for extracting, classifying, and exporting
skill position trajectories from Phase 1 JSON output files.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .data_loader import TrajectoryDataLoader, VideoTrajectories, ParsedTrajectory
from .offense_defense_classifier import (
    LineOfScrimmageClassifier, LineOfScrimmage,
    classify_offense_defense, fit_line_of_scrimmage
)
from .skill_position_filter import SkillPositionFilter, PlayerClassification

@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    # Input
    input_dir: Path = Path("data/viz_output")
    pattern: str = "*_trajectories.json"

    # Processing
    los_method: str = "minimal_movement"
    los_buffer: float = 100.0
    sideline_pct: float = 0.20
    min_detections: int = 30

    # Output
    output_dir: Path = Path("data/phase2_output")
    save_classified: bool = True
    save_skill_only: bool = True
    save_statistics: bool = True

@dataclass
class ExtractionResult:
    """Result of extracting trajectories from a single video."""
    video_name: str
    total_trajectories: int
    offense_count: int
    defense_count: int
    skill_position_count: int

    # LOS info
    los_x: float = 0.0
    los_confidence: float = 0.0

    # Skill position trajectories (for export)
    skill_trajectories: List[Dict] = None

    def __post_init__(self):
        if self.skill_trajectories is None:
            self.skill_trajectories = []


class SkillPositionExtractionPipeline:
    """
    Complete pipeline for extracting and classifying skill position trajectories.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.results: List[ExtractionResult] = []

    def run(self) -> List[ExtractionResult]:
        """Run the complete extraction pipeline."""
        print("=" * 60)
        print("NFL Route Tracker - Phase 2 Extraction Pipeline")
        print("=" * 60)

        # Load data
        loader = TrajectoryDataLoader()
        print(f"\nLoading trajectories from {self.config.input_dir}...")
        video_trajectories = loader.load_from_viz_output(self.config.input_dir)
        print(f"      Loaded {len(video_trajectories)} video files")

        if not video_trajectories:
            print("ERROR: No trajectory files found!")
            return []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each video
        print(f"\nClassifying offense/defense (method: {self.config.los_method})...")
        print(f"\nFiltering skill positions...")

        for vt in video_trajectories:
            result = self._process_video(vt)
            self.results.append(result)

        # Save results
        print(f"\nSaving results...")
        self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _process_video(self, vt: VideoTrajectories) -> ExtractionResult:
        """Process a single video's trajectories."""
        video_width = vt.metadata.get('video_width', 1920)
        video_height = vt.metadata.get('video_height', 1080)

        # Get trajectories as list
        trajectories = vt.trajectories

        # Step 1: Fit LOS and classify offense/defense
        classifier = LineOfScrimmageClassifier(method=self.config.los_method)
        classifier.fit(trajectories, video_width, video_height)

        # Classify all trajectories
        for traj in trajectories:
            traj.team = classifier.classify(traj)

        los = classifier.los

        # Step 2: Filter skill positions
        filter_obj = SkillPositionFilter(los=los, los_buffer=self.config.los_buffer,
                                        sideline_pct=self.config.sideline_pct,
                                        min_detections=self.config.min_detections)
        skill_positions, non_skill = filter_obj.filter_skill_positions(trajectories)

        # Create result
        result = ExtractionResult(
            video_name=vt.video_name,
            total_trajectories=len(trajectories),
            offense_count=len(classifier.get_offense_trajectories(trajectories)),
            defense_count=len(classifier.get_defense_trajectories(trajectories)),
            skill_position_count=len(skill_positions),
            los_x=los.x_value or (los.slope * video_height / 2 + los.intercept),
            los_confidence=los.confidence,
            skill_trajectories=[t.to_dict() for t in skill_positions]
        )

        print(f"  - {vt.video_name}: {len(skill_positions)} skill positions ")

        # Save individual video results if configured
        if self.config.save_classified:
            self._save_video_result(vt.video_name, trajectories, los)

        if self.config.save_skill_only and skill_positions:
            self._save_skill_positions(vt.video_name, skill_positions)

        return result

    def _save_video_result(self, video_name: str, trajectories: List[ParsedTrajectory],
                          los: LineOfScrimmage):
        """Save classified trajectories for a single video."""
        output_path = self.config.output_dir / f"{video_name}_classified.json"

        data = {'video_name': video_name,
                'los': {'x_value': los.x_value,
                        'slope': los.slope,
                        'intercept': los.intercept,
                        'confidence': los.confidence},
                'trajectories': [t.to_dict() for t in trajectories]}

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_skill_positions(self, video_name: str,
                             skill_trajectories: List[ParsedTrajectory]):
        """Save only skill position trajectories."""
        output_path = self.config.output_dir / f"{video_name}_skill_positions.json"

        data = {
            'video_name': video_name,
            'count': len(skill_trajectories),
            'trajectories': [t.to_dict() for t in skill_trajectories]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_results(self):
        """Save overall pipeline results and statistics."""
        if not self.results:
            return

        # Save statistics
        if self.config.save_statistics:
            stats = {
                'num_videos_processed': len(self.results),
                'total_trajectories': sum(r.total_trajectories for r in self.results),
                'total_skill_positions': sum(r.skill_position_count for r in self.results),
                'per_video': [asdict(r) for r in self.results]
            }

            stats_path = self.config.output_dir / "extraction_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

        # Save all skill positions combined
        all_skill = []
        for r in self.results:
            for t in r.skill_trajectories:
                t['source_video'] = r.video_name
                all_skill.append(t)

        if all_skill:
            combined_path = self.config.output_dir / "all_skill_positions.json"
            with open(combined_path, 'w') as f:
                json.dump({'total_count': len(all_skill), 'trajectories': all_skill}, f, indent=2)

    def _print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Videos processed:    {len(self.results)}")
        print(f"Total trajectories: {sum(r.total_trajectories for r in self.results)}")
        print(f"Total skill pos:    {sum(r.skill_position_count for r in self.results)}")
        print(f"\nOutput saved to: {self.config.output_dir}")
        print("=" * 60)


def run_pipeline(input_dir: str = "data/viz_output", output_dir: str = "data/phase2_output",
                los_method: str = "minimal_movement", los_buffer: float = 100.0,
                sideline_pct: float = 0.20, min_detections: int = 30,
                save_classified: bool = True, save_skill_only: bool = True) -> List[ExtractionResult]:
    """
    Convenience function to run the pipeline with custom parameters.
    """
    config = PipelineConfig(input_dir=Path(input_dir),
                            output_dir=Path(output_dir),
                            los_method=los_method,
                            los_buffer=los_buffer,
                            sideline_pct=sideline_pct,
                            min_detections=min_detections,
                            save_classified=save_classified,
                            save_skill_only=save_skill_only,)

    pipeline = SkillPositionExtractionPipeline(config)
    return pipeline.run()


# CLI interface
def main():
    parser = argparse.ArgumentParser(description="NFL Route Tracker - Phase 2 Pipeline")
    parser.add_argument("--input-dir", default="data/viz_output", help="Input directory with trajectory JSON files")
    parser.add_argument("--output-dir", default="data/phase2_output", help="Output directory for results")
    parser.add_argument("--los-method", default="minimal_movement",
                       choices=["clustering", "histogram", "vertical", "minimal_movement"],
                       help="Method for Line of Scrimmage detection")
    parser.add_argument("--los-buffer", type=float, default=100.0,
                       help="LOS buffer for skill position detection (pixels)")
    parser.add_argument("--sideline-pct", type=float, default=0.20,
                       help="Sideline percentage threshold")
    parser.add_argument("--min-detections", type=int, default=30,
                       help="Minimum detections for valid trajectory")
    parser.add_argument("--no-save-classified", action="store_true",
                       help="Don't save classified trajectories")
    parser.add_argument("--no-save-skill", action="store_true",
                       help="Don't save skill position only")

    args = parser.parse_args()

    run_pipeline(input_dir=args.input_dir,
                 output_dir=args.output_dir,
                 los_method=args.los_method,
                 los_buffer=args.los_buffer,
                 sideline_pct=args.sideline_pct,
                 min_detections=args.min_detections,
                 save_classified=not args.no_save_classified,
                 save_skill_only=not args.no_save_skill,)


if __name__ == "__main__":
    main()
