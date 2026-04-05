#!/usr/bin/env python3
"""
NFL Route Tracker Version 4 - ByteTrack Pipeline Demo
=====================================================

This demo showcases the ByteTrack-based pipeline with major performance improvements.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_route_tracker.core.config import get_pipeline_config, DetectionTrackerConfig
from nfl_route_tracker.tracking import DetectionTracker
from nfl_route_tracker.visualizations.visualizer import TrajectoryVisualizer


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_config_summary(config: DetectionTrackerConfig) -> None:
    """Print configuration summary."""
    print("Pipeline Configuration:")
    print("-" * 40)
    print(f"  YOLO Model: {config.detector_config.model_name}")
    print(f"  Input Size: {config.detector_config.imgsz}")
    print()
    print("  ByteTrack Parameters:")
    print(f"    track_high_thresh: {config.tracker_config.track_high_thresh}")
    print(f"    track_low_thresh: {config.tracker_config.track_low_thresh}")
    print(f"    track_buffer: {config.tracker_config.track_buffer}")
    print(f"    match_thresh: {config.tracker_config.match_thresh}")
    print(f"    fuse_score: {config.tracker_config.fuse_score}")
    print(f"    gmc_method: {config.tracker_config.gmc_method}")
    print()
    print("  NFL Filter:")
    print(f"    Min/Max Area: {config.nfl_filter_config.min_area} - {config.nfl_filter_config.max_area}")
    print(f"    Merge Overlaps: {config.nfl_filter_config.merge_overlaps}")
    print()
    print("  Trajectory Filtering:")
    print(f"    Min Length: {config.tracker_config.min_trajectory_length} detections")
    print()


def process_video(video_path: Path, output_dir: Path,
                  config: DetectionTrackerConfig,  max_frames: Optional[int] = None,
                  filter_trajectories: bool = True) -> bool:
    """
    Process a single video file with ByteTrack.
    """
    print(f"\nProcessing: {video_path.name}")

    # Create output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}_tracked.mp4"
    output_json = output_dir / f"{video_path.stem}_trajectories.json"
    output_plot = output_dir / f"{video_path.stem}_trajectories.png"

    # Initialize pipeline
    pipeline = DetectionTracker(config)

    try:
        # Process video
        print_header("Processing Video with ByteTrack")
        store = pipeline.process_video(
            str(video_path),
            output_video_path=str(output_video),
            output_json_path=str(output_json),
            max_frames=max_frames,
            filter_short_trajectories=filter_trajectories
        )

        # Print results
        print("\n" + "-" * 50)
        print("RESULTS:")
        print("-" * 50)
        print(f"  Trajectories found: {store.num_trajectories}")
        print(f"  Total detections: {store.total_detections}")

        # Print statistics
        stats = pipeline.get_statistics()
        print("\nPerformance:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Total time: {stats['total_processing_time']:.2f}s")
        print(f"  Average FPS: {stats['average_fps']:.2f}")

        # Visualize trajectories
        if store.num_trajectories > 0:
            print("\nGenerating trajectory plot...")
            viz = TrajectoryVisualizer(figsize=(14, 8))
            viz.plot_trajectories(store, output_path=str(output_plot), title=f"Tracked Trajectories - {video_path.stem}")

        # List outputs
        print("\n" + "-" * 50)
        print("OUTPUT FILES:")
        print("-" * 50)
        if output_video.exists():
            print(f"  Video: {output_video}")
        if output_json.exists():
            print(f"  JSON: {output_json}")
        if output_plot.exists():
            print(f"  Plot: {output_plot}")

        return True

    except Exception as e:
        print(f"\nERROR processing {video_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def batch_process(video_dir: Path, output_dir: Path, config: DetectionTrackerConfig) -> int:
    """
    Process multiple videos in a directory.
    """
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return 0

    # Sort by name
    video_files.sort()

    print(f"\nFound {len(video_files)} videos to process")
    print("=" * 70)

    # Process each video
    success_count = 0
    failed_videos = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {video_path.name}")

        # Create subdirectory for this video
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        if process_video(video_path, video_output_dir, config):
            success_count += 1
        else:
            failed_videos.append(video_path.name)

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Successful: {success_count}/{len(video_files)}")
    print(f"  Failed: {len(failed_videos)}")

    if failed_videos:
        print("\nFailed videos:")
        for name in failed_videos:
            print(f"  - {name}")

    return success_count

def main():
    """Main entry point."""
    import time

    parser = argparse.ArgumentParser(description = "NFL Route Tracker v4 - Demo",
                                    formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('video', nargs='?', default=None, help='Path to video file (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: data/viz_output/)')
    parser.add_argument('--max-frames', type=int, default=None, help='Limit frames per video (for quick testing)')
    parser.add_argument('--batch', action='store_true', help='Process all videos in data/video_test/')

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    video_test_dir = project_root / "data" / "video_test"
    default_output_dir = project_root / "data" / "viz_output"

    output_dir = Path(args.output) if args.output else default_output_dir

    # Get configuration based on preset
    print_header("NFL Route Tracker - ByteTrack Pipeline")
    print("Loading configuration from config.py...")

    config = get_pipeline_config()

    print_config_summary(config)

    # Determine what to process
    if args.batch:
        print_header("Batch Processing Mode")
        success_count = batch_process(video_test_dir, output_dir, config)
    elif args.video:
        # Use specified video
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = project_root / args.video

        if not video_path.exists():
            print(f"\nERROR: Video not found: {video_path}")
            sys.exit(1)

        print_header("Single Video Mode")
        success = process_video(video_path, output_dir, config, args.max_frames)

        if success:
            print("\nDone!")
        else:
            sys.exit(1)

    else:
        # Find first video in test directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_test_dir.glob(f"*{ext}"))

        if not video_files:
            print(f"\nNo videos found in {video_test_dir}")
            print("\nPlace video files in data/video_test/ or specify a video path.")
            sys.exit(1)

        # Sort and take first
        video_files.sort()
        video_path = video_files[0]

        print_header("Single Video Mode (auto-selected)")
        print(f"Using: {video_path.name}")
        print(f"Output directory: {output_dir}")
        print()

        success = process_video(video_path, output_dir, config, args.max_frames)

        if success:
            print("\nDone!")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()