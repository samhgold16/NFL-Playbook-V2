#!/usr/bin/env python3
"""
NFL Route Tracker - Detection Pipeline Diagnostic Tool
=====================================================

This script helps identify WHERE in the filtering pipeline detections are being lost.
It shows the count at each stage so you can see exactly which filter is too aggressive.

Usage:
    python diagnostic_debug.py [video_path]

Output:
    - Side-by-side visualization of each filtering stage
    - Statistics showing drop-off at each stage
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nfl_route_tracker.core.config import get_default_pipeline_config
from nfl_route_tracker.core.video_loader import VideoLoader
from nfl_route_tracker.detection.player_detector import PlayerDetector, DetectionResult
from nfl_route_tracker.detection.nfl_filter import NFLDetectionFilter
from nfl_route_tracker.tracking.temporal_aggregator import TemporalAggregator


def draw_detections(frame, detections, label, color=(0, 255, 0), show_conf=False):
    """Draw detections on frame with label."""
    frame_copy = frame.copy()
    for i, det in enumerate(detections):
        x, y = int(det.x), int(det.y)
        w, h = int(det.width), int(det.height)

        # Draw box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)

        # Draw label
        text = f"{label}:{i}"
        if show_conf:
            text += f" {det.confidence:.2f}"

        cv2.putText(frame_copy, text, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add count in corner
    cv2.putText(frame_copy, f"{label}: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame_copy


def get_filter_stats(det, filter_config, frame_height=984):
    """Get statistics for a detection to understand why it passed/failed."""
    area = det.area
    aspect = det.width / det.height if det.height > 0 else 0
    center_y = det.y + det.height / 2

    stats = {
        'confidence': det.confidence,
        'area': area,
        'aspect': aspect,
        'center_y': center_y,
        'y': det.y,
    }

    # Determine zone
    if center_y >= filter_config.near_y_threshold:
        stats['zone'] = 'near'
        stats['zone_area_range'] = filter_config.near_area_range
        stats['zone_aspect_range'] = filter_config.near_aspect_range
    elif center_y <= filter_config.far_y_threshold:
        stats['zone'] = 'far'
        stats['zone_area_range'] = filter_config.far_area_range
        stats['zone_aspect_range'] = filter_config.far_aspect_range
    else:
        stats['zone'] = 'mid'
        stats['zone_area_range'] = filter_config.mid_area_range
        stats['zone_aspect_range'] = filter_config.mid_aspect_range

    return stats


def diagnose_video(video_path, max_frames=150, output_path=None):
    """
    Run diagnostic on a video to see where detections are lost.

    Args:
        video_path: Path to video file
        max_frames: Number of frames to analyze
        output_path: Optional path for output video
    """
    print("=" * 70)
    print("NFL ROUTE TRACKER - DETECTION PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # Load config and components
    config = get_default_pipeline_config()

    # Override with LENIENT settings based on diagnostic findings
    # The key insight: raw YOLO outperforms filtered output!
    # Let DeepSORT handle false positives via appearance matching
    config.nfl_filter_config.min_confidence = 0.05  # Very lenient
    config.nfl_filter_config.min_area = 150  # Catch tiny players
    config.nfl_filter_config.max_area = 35000  # Catch huge players
    config.nfl_filter_config.min_aspect_ratio = 0.10  # Even more lenient for crouching
    config.nfl_filter_config.max_aspect_ratio = 1.5  # Allow players facing camera (wider than tall)
    # MERGE: Set to 0.5 to only merge very highly overlapping detections
    # NFL players overlap naturally - don't merge them, let DeepSORT handle
    config.nfl_filter_config.merge_iou_threshold = 0.5  # High = keep more overlapping players
    config.nfl_filter_config.near_area_range = (1500, 35000)
    config.nfl_filter_config.far_area_range = (150, 15000)
    config.nfl_filter_config.mid_area_range = (500, 25000)

    # Temporal - DISABLE for now since it's causing problems
    # The clustering approach is causing bouncing and missed detections
    # Let DeepSORT handle stability instead
    config.temporal_config.enabled = True  # ENABLED - user reports TEMPORAL is worst
    config.temporal_config.min_detection_count = 1

    # Camera - disable for diagnostic
    config.camera_config.enabled = True

    detector = PlayerDetector(config.detector_config)
    nfl_filter = NFLDetectionFilter(
        min_area=config.nfl_filter_config.min_area,
        max_area=config.nfl_filter_config.max_area,
        min_aspect_ratio=config.nfl_filter_config.min_aspect_ratio,
        max_aspect_ratio=config.nfl_filter_config.max_aspect_ratio,
        min_confidence=config.nfl_filter_config.min_confidence,
        low_confidence=config.nfl_filter_config.low_confidence,
        near_y_threshold=config.nfl_filter_config.near_y_threshold,
        far_y_threshold=config.nfl_filter_config.far_y_threshold,
        near_area_range=config.nfl_filter_config.near_area_range,
        near_aspect_range=config.nfl_filter_config.near_aspect_range,
        far_area_range=config.nfl_filter_config.far_area_range,
        far_aspect_range=config.nfl_filter_config.far_aspect_range,
        mid_area_range=config.nfl_filter_config.mid_area_range,
        mid_aspect_range=config.nfl_filter_config.mid_aspect_range,
        min_y_position=config.nfl_filter_config.min_y_position,
        max_y_position=config.nfl_filter_config.max_y_position
    )
    temporal = TemporalAggregator(
        window_size=config.temporal_config.window_size,
        stride=config.temporal_config.stride,
        method=config.temporal_config.aggregation_method,
        confidence_weight=config.temporal_config.confidence_weight,
        min_detection_count=config.temporal_config.min_detection_count,
        position_threshold=config.temporal_config.position_threshold
    )

    # Statistics tracking
    stats = defaultdict(int)
    filter_fail_reasons = defaultdict(int)

    print(f"\nAnalyzing: {video_path}")
    print(f"Max frames: {max_frames}")
    print(f"\nLENIENT CONFIG being used (v5 - based on diagnostic findings):")
    print(f"  Confidence threshold: {config.nfl_filter_config.min_confidence}")
    print(f"  Area range: {config.nfl_filter_config.min_area} - {config.nfl_filter_config.max_area}")
    print(f"  Aspect ratio: {config.nfl_filter_config.min_aspect_ratio} - {config.nfl_filter_config.max_aspect_ratio}")
    print(f"  Near zone (y>={config.nfl_filter_config.near_y_threshold}): area {config.nfl_filter_config.near_area_range}")
    print(f"  Far zone (y<={config.nfl_filter_config.far_y_threshold}): area {config.nfl_filter_config.far_area_range}")
    print(f"  Mid zone: area {config.nfl_filter_config.mid_area_range}")
    print(f"  Merge IOU threshold: {config.nfl_filter_config.merge_iou_threshold}")

    # Setup video writer if output requested
    video_writer = None
    if output_path:
        with VideoLoader(video_path) as loader:
            h, w = loader.metadata.height, loader.metadata.width
            # 5 columns: Raw, Confidence, Area, Zone, Temporal
            out_w = w * 3
            out_h = h * 2  # Make it 3x2 instead of 1x5 for better aspect ratio
            # Use 'avc1' codec for better compatibility, fallback to 'mp4v'
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(output_path, fourcc, loader.metadata.fps, (out_w, out_h))
                if not video_writer.isOpened():
                    raise Exception("avc1 failed")
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, loader.metadata.fps, (out_w, out_h))
                    if not video_writer.isOpened():
                        raise Exception("mp4v failed")
                except Exception as e:
                    print(f"WARNING: Could not create video writer: {e}")
                    print("         Continuing without video output...")
                    video_writer = None

    # Process frames
    with VideoLoader(video_path) as loader:
        frame_height = loader.metadata.height

        for frame_id, frame in loader:
            if frame_id >= max_frames:
                break

            if frame_id % 30 == 0:
                print(f"\nProcessing frame {frame_id}...")

            # STAGE 1: Raw YOLO detections
            raw_detections = detector.detect(frame)
            stats['raw'] += len(raw_detections)

            if frame_id % 30 == 0:
                print(f"  Raw detections: {len(raw_detections)}")

            # STAGE 2: NFL Filter - Confidence
            conf_filtered = [d for d in raw_detections if d.confidence >= nfl_filter.min_confidence]
            stats['after_confidence_filter'] += len(conf_filtered)
            for d in raw_detections:
                if d.confidence < nfl_filter.min_confidence:
                    filter_fail_reasons['confidence'] += 1

            # STAGE 3: NFL Filter - Area
            area_filtered = []
            for d in conf_filtered:
                if nfl_filter.min_area <= d.area <= nfl_filter.max_area:
                    area_filtered.append(d)
                else:
                    filter_fail_reasons[f'area_{get_filter_stats(d, nfl_filter)["zone"]}'] += 1
            stats['after_area_filter'] += len(area_filtered)

            # STAGE 4: NFL Filter - Aspect Ratio
            aspect_filtered = []
            for d in area_filtered:
                aspect = d.width / d.height if d.height > 0 else 0
                if nfl_filter.min_aspect_ratio <= aspect <= nfl_filter.max_aspect_ratio:
                    aspect_filtered.append(d)
                else:
                    filter_fail_reasons['aspect_ratio'] += 1
            stats['after_aspect_filter'] += len(aspect_filtered)

            # STAGE 5: NFL Filter - Y Position
            ypos_filtered = []
            for d in aspect_filtered:
                if nfl_filter.min_y_position <= d.y <= nfl_filter.max_y_position:
                    ypos_filtered.append(d)
                else:
                    filter_fail_reasons['y_position'] += 1
            stats['after_ypos_filter'] += len(ypos_filtered)

            # STAGE 6: NFL Filter - Field Zone (this is part of main filter)
            # The _check_field_zone is called within filter_detections, not standalone
            # So zone_filtered == ypos_filtered at this point (zone check was already done if we used filter_detections)
            # For diagnostic, we'll skip explicit zone filtering here
            zone_filtered = ypos_filtered
            stats['after_zone_filter'] += len(zone_filtered)

            # STAGE 7: Merge Overlapping (O-line)
            merged = nfl_filter.merge_overlapping_detections(
                zone_filtered,
                iou_threshold=config.nfl_filter_config.merge_iou_threshold
            )
            stats['after_merge'] += len(merged)

            # STAGE 8: NMS (Non-Maximum Suppression) - DISABLED FOR NFL!
            # NMS is fundamentally WRONG for NFL because:
            # - We have 22 DIFFERENT players close together
            # - Their bounding boxes naturally overlap (O-line, defensive line)
            # - NMS suppresses one player even if it's a different person
            # - This explains why 69% of detections were being removed!
            #
            # SOLUTION: Disable NMS entirely - let DeepSORT handle
            # appearance matching to distinguish between players
            nms_filtered = merged  # Pass through directly - NO NMS
            stats['after_nms'] += len(nms_filtered)

            # STAGE 9: Temporal Aggregation (DISABLED - causing problems)
            # User reports TEMPORAL video is worst - boxes bounce and miss players
            # Let DeepSORT handle stability instead
            if config.temporal_config.enabled:
                temporal.add_frame(frame_id, nms_filtered)
                temporal_result = temporal.get_aggregated(frame_id)
            else:
                temporal_result = nms_filtered  # Pass through directly
            stats['after_temporal'] += len(temporal_result)

            # Generate visualization every 30 frames
            if frame_id % 30 == 0 and video_writer:
                # Create 5-column visualization
                col1 = draw_detections(frame, raw_detections, "RAW", (0, 255, 0))
                col2 = draw_detections(frame, conf_filtered, "CONF", (255, 255, 0))
                col3 = draw_detections(frame, area_filtered, "AREA", (0, 255, 255))
                col4 = draw_detections(frame, merged, "MERGED", (255, 0, 255))
                col5 = draw_detections(frame, temporal_result, "TEMPORAL", (128, 128, 255))
                # making sixth, empty column to make a 3 x 2 output instead of 1 x 5
                empty_col = np.zeros_like(frame)

                row1 = np.hstack([col1, col2, col3])
                row2 = np.hstack([col4, col5, empty_col])
                combined = np.vstack([row1, row2])
                video_writer.write(combined)

    # Close video writer
    if video_writer:
        video_writer.release()

    # Print final statistics
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)

    print("\nDetection counts (cumulative):")
    print(f"  Raw YOLO detections:           {stats['raw']:6d}")
    print(f"  After confidence filter:       {stats['after_confidence_filter']:6d}  "
          f"(-{stats['raw'] - stats['after_confidence_filter']})")
    print(f"  After area filter:            {stats['after_area_filter']:6d}  "
          f"(-{stats['after_confidence_filter'] - stats['after_area_filter']})")
    print(f"  After aspect ratio filter:     {stats['after_aspect_filter']:6d}  "
          f"(-{stats['after_area_filter'] - stats['after_aspect_filter']})")
    print(f"  After y-position filter:      {stats['after_ypos_filter']:6d}  "
          f"(-{stats['after_aspect_filter'] - stats['after_ypos_filter']})")
    print(f"  After zone filter:            {stats['after_zone_filter']:6d}  "
          f"(-{stats['after_ypos_filter'] - stats['after_zone_filter']})")
    print(f"  After merge:                  {stats['after_merge']:6d}  "
          f"(-{stats['after_zone_filter'] - stats['after_merge']})")
    print(f"  After NMS:                   {stats['after_nms']:6d}  "
          f"(-{stats['after_merge'] - stats['after_nms']})")
    print(f"  After temporal aggregation:   {stats['after_temporal']:6d}  "
          f"(-{stats['after_nms'] - stats['after_temporal']})")

    print(f"\nFilter failure breakdown:")
    total_failures = sum(filter_fail_reasons.values())
    for reason, count in sorted(filter_fail_reasons.items(), key=lambda x: -x[1]):
        pct = count / total_failures * 100 if total_failures > 0 else 0
        print(f"  {reason:30s}: {count:6d} ({pct:5.1f}%)")

    # Calculate average detections per frame
    avg_raw = stats['raw'] / max_frames if max_frames > 0 else 0
    avg_final = stats['after_temporal'] / max_frames if max_frames > 0 else 0

    print(f"\nAverage detections per frame:")
    print(f"  Raw:       {avg_raw:.1f}")
    print(f"  Final:     {avg_final:.1f}")
    print(f"  Retention: {avg_final/avg_raw*100:.1f}%" if avg_raw > 0 else "  N/A")

    if output_path:
        print(f"\nOutput video saved to: {output_path}")

    return stats, filter_fail_reasons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFL Detection Pipeline Diagnostic")
    parser.add_argument('video', nargs='?', default=None, help='Video path')
    parser.add_argument('--max-frames', type=int, default=150, help='Max frames to analyze')
    parser.add_argument('--output', type=str, default=None, help='Output video path')

    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        # Default to first video in test folder
        video_path = Path(__file__).parent / "data" / "video_test"
        videos = list(video_path.glob("*.mp4"))
        if videos:
            video_path = videos[0]
        else:
            print("ERROR: No video found. Please specify a video path.")
            sys.exit(1)

    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    output_path = args.output or str(Path(video_path).parent / f"{Path(video_path).stem}_diagnostic.mp4")

    diagnose_video(str(video_path), max_frames=args.max_frames, output_path=output_path)
