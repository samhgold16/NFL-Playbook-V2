#!/usr/bin/env python3
"""
NFL Route Tracker Version 3 - Complete Pipeline Demo
===================================
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_route_tracker.core.config import (
    get_default_pipeline_config,
    DetectionTrackerConfig,
    DetectorConfig,
    TrackerConfig,
    NFLDetectionFilterConfig,
    TemporalAggregatorConfig,
    CameraStabilizerConfig
)
from nfl_route_tracker.tracking.detection_tracker import DetectionTracker
from nfl_route_tracker.visualizations.visualizer import TrajectoryVisualizer

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_config_summary(config: DetectionTrackerConfig) -> None:
    """Print configuration summary."""
    print("Pipeline Configuration:")
    print(f"  YOLO Model: {config.detector_config.model_name}")
    print(f"  Confidence Threshold: {config.detector_config.confidence_threshold}")
    print(f"  Input Size: {config.detector_config.imgsz}")
    print()
    print("  NFL Filter:")
    print(f"    Min/Max Area: {config.nfl_filter_config.min_area} - {config.nfl_filter_config.max_area}")
    print(f"    Min/Max Aspect Ratio: {config.nfl_filter_config.min_aspect_ratio} - {config.nfl_filter_config.max_aspect_ratio}")
    print(f"    Merge Overlaps: {config.nfl_filter_config.merge_overlaps}")
    print()
    print("  Temporal Aggregation:")
    print(f"    Enabled: {config.temporal_config.enabled}")
    print(f"    Window Size: {config.temporal_config.window_size} frames")
    print(f"    Method: {config.temporal_config.aggregation_method}")
    print()
    print("  DeepSORT Tracker:")
    print(f"    Max Age: {config.tracker_config.max_age}")
    print(f"    N Init: {config.tracker_config.n_init}")
    print(f"    Embedder: {config.tracker_config.embedder}")
    print()
    print("  Camera Stabilization (Phase 1B):")
    print(f"    Enabled: {config.camera_config.enabled}")
    if config.camera_config.enabled:
        print(f"    Feature Method: {config.camera_config.feature_method}")
        print(f"    Max Features: {config.camera_config.max_features}")
        print(f"    Smoothing Window: {config.camera_config.smoothing_window} frames")
        print(f"    RANSAC Threshold: {config.camera_config.ransac_threshold} px")


import cv2
from nfl_route_tracker.tracking.camera_stabilizer import CameraStabilizer, visualize_stabilization
from nfl_route_tracker.core.video_loader import VideoLoader

def process_single_video(video_path: Path,
                        output_dir: Path,
                        config: DetectionTrackerConfig,
                        max_frames: Optional[int] = None,
                        debug_stabilization: bool = True,
                        debug_frames: int = 150) -> bool:
    """
    Process a single video file.
    """
    print(f"\nProcessing: {video_path.name}")

    # Create output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}_tracked.mp4"
    output_json = output_dir / f"{video_path.stem}_trajectories.json"
    output_plot = output_dir / f"{video_path.stem}_trajectories.png"
    output_debug = output_dir / f"{video_path.stem}_stabilization_debug.mp4"


    # Initialize pipeline
    pipeline = DetectionTracker(config)

    if debug_stabilization and config.camera_config.enabled:
        print(f"\n  [Debug] Running stabilization visualization ({debug_frames} frames)...")

        with VideoLoader(video_path) as loader:
            meta = loader.metadata
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(
                str(output_debug), fourcc, meta.fps, (meta.width * 2, meta.height)
            )
            debug_stabilizer = CameraStabilizer(config.camera_config)

            for frame_num, frame in loader:
                if frame_num >= debug_frames:
                    break
                debug_stabilizer.update(frame)
                raw_dets        = pipeline._detector.detect(frame)
                stabilized_dets = debug_stabilizer.stabilize_detections(raw_dets)
                debug_writer.write(visualize_stabilization(frame, raw_dets, stabilized_dets))

            debug_writer.release()

        print(f"  [Debug] Saved → {output_debug}")

    try:
        # Process video
        store = pipeline.process_video(str(video_path), output_video_path = str(output_video),
                                        output_json_path = str(output_json), max_frames = max_frames)

        # Print results
        print("\n" + "-" * 50)
        print("RESULTS:")
        print("-" * 50)
        print(f"  Trajectories found: {store.num_trajectories}")
        print(f"  Total detections: {store.total_detections}")

        # Print statistics
        stats = pipeline.get_statistics()
        print("\nProcessing Statistics:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Total time: {stats['total_processing_time']:.2f}s")
        print(f"  Average FPS: {stats['average_fps']:.2f}")
        print(f"  Raw detections: {stats['raw_detections']}")
        print(f"  Filtered detections: {stats['filtered_detections']}")
        print(f"  Filter ratio: {stats['filter_ratio']:.2%}")

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

# dont use until confimring that one video works
def batch_process(video_dir: Path, output_dir: Path, config: DetectionTrackerConfig, max_videos: Optional[int] = None) -> int:
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

    # Limit if requested
    if max_videos:
        video_files = video_files[:max_videos]

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

        if process_single_video(video_path, video_output_dir, config):
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
    parser = argparse.ArgumentParser(description = "NFL Route Tracker - Version 3Demo",
                                     formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('video', nargs='?', default=None, help='Path to video file (optional)')

    parser.add_argument('--batch', action='store_true', help='Process all videos in data/video_test/')

    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process in batch mode')

    parser.add_argument('--output', type=str, default=None, help='Output directory (default: data/viz_output/)')

    parser.add_argument('--no-video', action='store_true', help='Skip video output (faster processing)')

    parser.add_argument('--no-json', action='store_true', help='Skip JSON trajectory output')

    parser.add_argument('--max-frames', type=int, default=None, help='Limit frames per video (for quick testing)')

    # EXTRA ARGUMENTS TO CONSIDER IF WANTED

    # parser.add_argument(
    #     '--no-ghost-filter',
    #     action='store_true',
    #     help='Keep ghost boxes (Kalman predictions). Default: filter them out'
    # )

    # parser.add_argument(
    #     '--min-hits',
    #     type=int,
    #     default=1,
    #     help='Minimum detections before showing a track (default: 1)'
    # )
    # parser.add_argument(
    #     '--no-camera',
    #     action='store_true',
    #     help='Disable camera motion compensation (Phase 1B)'
    # )

    # parser.add_argument(
    #     '--camera-method',
    #     type=str,
    #     default='shi-tomasi',
    #     choices=['shi-tomasi', 'orb', 'sift'],
    #     help='Feature detection method for camera stabilization (default: shi-tomasi)'
    # )
    # parser.add_argument(
    #     '--camera-features',
    #     type=int,
    #     default=500,
    #     help='Maximum features to track for camera stabilization (default: 500)'
    # )

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    video_test_dir = project_root / "data" / "video_test"
    default_output_dir = project_root / "data" / "viz_output"

    output_dir = Path(args.output) if args.output else default_output_dir

    # Get configuration
    print_header("NFL Route Tracker")
    print("Loading configuration...")

    config = get_default_pipeline_config()

    # Override settings from command line
    config.save_video = not args.no_video
    config.save_trajectories = not args.no_json
    config.verbose = True
    config.progress_interval = 50

    print_config_summary(config)

    # Determine what to process
    if args.batch:
        # Batch mode
        print_header("Batch Processing Mode")
        success_count = batch_process(video_test_dir, output_dir, config, max_videos = args.max_videos)
        print(f"\nDone! Processed {success_count} videos.")

    elif args.video:
        # Single specific video
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = project_root / args.video

        if not video_path.exists():
            print(f"\nERROR: Video not found: {video_path}")
            sys.exit(1)

        print_header("Single Video Mode")
        success = process_single_video(video_path, output_dir, config, max_frames=args.max_frames)
        if success:
            print("\nDone!")
        else:
            sys.exit(1)

    else:
        # Default: find first video
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_test_dir.glob(f"*{ext}"))

        if not video_files:
            print(f"\nNo videos found in {video_test_dir}")
            sys.exit(1)

        # Sort and take first
        video_files.sort()
        video_path = video_files[0]

        print_header("Single Video Mode (auto-selected)")
        print(f"Using: {video_path.name}")
        print(f"Output directory: {output_dir}")

        success = process_single_video(video_path, output_dir, config, max_frames=args.max_frames)

        if success:
            print("\nDone!")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
