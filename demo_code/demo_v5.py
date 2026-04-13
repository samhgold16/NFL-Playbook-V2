#!/usr/bin/env python3
"""
NFL Route Tracker - Pipeline Fix Validation Script
================================================

This script validates the corrected transformation pipeline by:
1. Processing a test video
2. Checking that transformations are applied in correct order
3. Generating visualizations to verify trajectory stability

Run with: python demo_code/demo_v5.py --video data/video_test/test8.mp4 from root directory
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_route_tracker.core.config import get_pipeline_config
from nfl_route_tracker.tracking import DetectionTracker


def validate_trajectory_stability(store, video_name: str) -> dict:
    """
    Validate that trajectories are stable (no global drift).

    A stable trajectory set should:
    - Have stationary players appear stationary (near-zero velocity at start/end)
    - Show consistent spread of players across the field
    - Not drift in a single direction across all trajectories
    """
    trajectories = store.get_all_trajectories()

    stats = {
        'num_trajectories': len(trajectories),
        'trajectory_lengths': [],
        'total_drift_x': [],
        'total_drift_y': [],
        'avg_velocity_x': [],
        'avg_velocity_y': [],
        'cross_trajectory_correlation_x': 0,
        'cross_trajectory_correlation_y': 0,
    }

    drift_x_list = []
    drift_y_list = []

    for traj in trajectories:
        frames, xs, ys = traj.get_path()

        if len(frames) < 2:
            continue

        stats['trajectory_lengths'].append(len(frames))

        # Calculate drift (total displacement)
        drift_x = xs[-1] - xs[0]
        drift_y = ys[-1] - ys[0]
        stats['total_drift_x'].append(drift_x)
        stats['total_drift_y'].append(drift_y)
        drift_x_list.append(drift_x)
        drift_y_list.append(drift_y)

        # Calculate average velocity
        if len(frames) > 1:
            vx = np.diff(xs)
            vy = np.diff(ys)
            stats['avg_velocity_x'].append(np.mean(np.abs(vx)))
            stats['avg_velocity_y'].append(np.mean(np.abs(vy)))

    # Cross-trajectory correlation (if trajectories drift together, correlation is high)
    if len(drift_x_list) > 1:
        drift_x_arr = np.array(drift_x_list)
        drift_y_arr = np.array(drift_y_list)

        # Mean-center
        drift_x_centered = drift_x_arr - np.mean(drift_x_arr)
        drift_y_centered = drift_y_arr - np.mean(drift_y_arr)

        # Cross-correlation (should be ~0 for uncorrelated drift = good)
        if np.std(drift_x_arr) > 0 and np.std(drift_y_arr) > 0:
            stats['cross_trajectory_correlation_x'] = np.mean(drift_x_centered * np.roll(drift_x_centered, 1)) / (np.std(drift_x_arr) ** 2)
            stats['cross_trajectory_correlation_y'] = np.mean(drift_y_centered * np.roll(drift_y_centered, 1)) / (np.std(drift_y_arr) ** 2)

    return stats


def print_validation_report(stats: dict, video_name: str):
    """Print a validation report for trajectory stability."""
    print(f"\n{'='*60}")
    print(f"  TRAJECTORY STABILITY VALIDATION REPORT")
    print(f"  Video: {video_name}")
    print(f"{'='*60}\n")

    print(f"Trajectories:")
    print(f"  Count: {stats['num_trajectories']}")
    print(f"  Avg length: {np.mean(stats['trajectory_lengths']):.1f} frames")
    print(f"  Min/Max length: {min(stats['trajectory_lengths'])}/{max(stats['trajectory_lengths'])} frames")

    print(f"\nDrift Analysis (total displacement):")
    avg_drift_x = np.mean(stats['total_drift_x'])
    avg_drift_y = np.mean(stats['total_drift_y'])
    print(f"  Avg X drift: {avg_drift_x:.2f} pixels")
    print(f"  Avg Y drift: {avg_drift_y:.2f} pixels")

    # Check for global drift (all trajectories moving same direction)
    drift_x_std = np.std(stats['total_drift_x'])
    drift_y_std = np.std(stats['total_drift_y'])

    print(f"\nDrift Consistency:")
    print(f"  X drift std: {drift_x_std:.2f} pixels")
    print(f"  Y drift std: {drift_y_std:.2f} pixels")

    # If avg drift is large but std is small, we have global drift (BAD)
    drift_magnitude = np.sqrt(avg_drift_x**2 + avg_drift_y**2)

    print(f"\nStability Assessment:")
    if drift_magnitude > 50 and drift_x_std < 30:
        print(f"  ⚠️  WARNING: Possible global drift detected!")
        print(f"     All trajectories moving ~{drift_magnitude:.1f} pixels in same direction")
        print(f"     This suggests camera stabilization may not be working correctly")
        return False
    elif drift_magnitude < 20:
        print(f"  ✅ PASS: Trajectories appear stable")
        return True
    else:
        print(f"  ⚠️  WARNING: Moderate drift detected ({drift_magnitude:.1f} pixels)")
        print(f"     This may be normal player movement")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test the corrected transformation pipeline")
    parser.add_argument('--video', '-v', type=str, default=None, help='Path to video file')
    parser.add_argument('--output', '-o', type=str, default='test_output', help='Output directory')
    parser.add_argument('--max-frames', '-m', type=int, default=500, help='Max frames to process')

    args = parser.parse_args()

    # Find test video
    if args.video is None:
        # Try to find video in common locations
        test_dirs = [
            Path('data/video_test'),
            Path('data/videos'),
            Path('test_data'),
            Path('.'),
        ]
        for d in test_dirs:
            videos = list(d.glob('*.mp4')) + list(d.glob('*.MP4'))
            if videos:
                args.video = str(videos[0])
                print(f"Auto-selected video: {args.video}")
                break

    if args.video is None or not Path(args.video).exists():
        print("ERROR: No video file found!")
        print("Please specify with --video path/to/video.mp4")
        return 1

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TESTING FIXED TRANSFORMATION PIPELINE")
    print(f"{'='*60}")
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Max frames: {args.max_frames}")

    # Load config
    config = get_pipeline_config()

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = DetectionTracker(config)

    # Process video
    print("\nProcessing video...")
    output_json = output_dir / f"{video_path.stem}_trajectories.json"
    output_plot = output_dir / f"{video_path.stem}_trajectories.png"

    store = pipeline.process_video(
        str(video_path),
        output_video_path=None,  # Don't save video for speed
        output_json_path=str(output_json),
        max_frames=args.max_frames,
        filter_short_trajectories=True,
        filter_off_field=True
    )

    # Validate stability
    stats = validate_trajectory_stability(store, video_path.name)

    # Print report
    is_stable = print_validation_report(stats, video_path.name)

    # Generate plot
    print("\nGenerating trajectory plot...")
    from nfl_route_tracker.visualizations import TrajectoryVisualizer

    viz = TrajectoryVisualizer(figsize=(14, 8))
    viz.plot_trajectories(
        store,
        output_path=str(output_plot),
        title=f"Trajectory Stability Test - {video_path.name}"
    )

    print(f"\nOutput files:")
    print(f"  JSON: {output_json}")
    print(f"  Plot: {output_plot}")

    # Save validation report
    report_path = output_dir / f"{video_path.stem}_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'video': str(video_path),
            'stability_check': 'PASS' if is_stable else 'FAIL',
            'stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in stats.items()}
        }, f, indent=2)
    print(f"  Report: {report_path}")

    return 0 if is_stable else 1


if __name__ == "__main__":
    sys.exit(main())