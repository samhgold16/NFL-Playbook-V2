#!/usr/bin/env python3
"""
NFL Route Tracker - Comprehensive Transformation Comparison Test
================================================================

This script compares the basic rotation transform vs the field axis projection
transform to demonstrate which one correctly preserves spatial relationships.

KEY TEST CASE:
- Players 'a1' and 'a2' are on the SAME yardline (slanted in video)
- After correct transformation, their X coordinates should be nearly identical
- A simple rotation does NOT guarantee this; field axis projection DOES
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nfl_route_tracker.tracking.field_axis_transform import (
    FieldAxisProjectionTransform,
    RotatedFieldAxisTransform
)
from nfl_route_tracker.tracking.field_orientation_detector import FieldOrientationDetector
from nfl_route_tracker.core.video_loader import VideoLoader


def compare_transforms(video_path: str = None):
    """
    Compare basic rotation vs field axis projection.
    """
    print("="*70)
    print("TRANSFORMATION COMPARISON TEST")
    print("="*70)

    # Parameters from the diagnostic
    yard_line_angle = -2.9  # degrees (from diagnostic)
    rotation_angle = 2.9   # degrees (negative of yard line angle)
    center_x, center_y = 960, 492

    # Test points simulating the user's scenario
    # Two receivers on the same yardline (in video coords)
    same_yardline_players = [
        {'id': 7, 'x': 1239.8, 'y': 301.5},   # Track 7 first position
        {'id': 10, 'x': 1350.9, 'y': 351.3},  # Track 10 first position
        {'id': 13, 'x': 1285.2, 'y': 270.8},  # Track 13 first position
    ]

    # Player on different yardline
    diff_yardline_player = {'id': 14, 'x': 1225.9, 'y': 809.9}  # Track 14

    print("\n--- Input Positions (Video Coordinates) ---")
    print("\nPlayers at TOP of video (expected to be on same general yardline area):")
    for p in same_yardline_players:
        print(f"  ID {p['id']:2d}: x={p['x']:7.1f}, y={p['y']:7.1f}")

    y_vals = [p['y'] for p in same_yardline_players]
    print(f"  Y spread: {max(y_vals)-min(y_vals):.1f} pixels (~{(max(y_vals)-min(y_vals))/19:.1f} yards depth)")

    print(f"\nPlayer at BOTTOM of video (expected on different yardline):")
    print(f"  ID {diff_yardline_player['id']:2d}: x={diff_yardline_player['x']:7.1f}, y={diff_yardline_player['y']:7.1f}")

    # =========================================================================
    # Transform 1: Basic Rotation (current implementation)
    # =========================================================================
    print("\n" + "="*70)
    print("TRANSFORM 1: Basic Rotation (current implementation)")
    print("="*70)

    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    def rotate_point(x, y):
        """Standard rotation around center."""
        dx = x - center_x
        dy = y - center_y
        rx = dx * cos_a - dy * sin_a + center_x
        ry = dx * sin_a + dy * cos_a + center_y
        return rx, ry

    print("\nAfter basic rotation:")
    for p in same_yardline_players:
        rx, ry = rotate_point(p['x'], p['y'])
        print(f"  ID {p['id']:2d}: x={rx:7.1f}, y={ry:7.1f}")

    x_vals_rotated = [rotate_point(p['x'], p['y'])[0] for p in same_yardline_players]
    print(f"  X spread: {max(x_vals_rotated)-min(x_vals_rotated):.1f} pixels")

    rx14, ry14 = rotate_point(diff_yardline_player['x'], diff_yardline_player['y'])
    print(f"\n  ID {diff_yardline_player['id']:2d}: x={rx14:7.1f}, y={ry14:7.1f}")

    # Check horizontal separation between top players and bottom player
    avg_top_x_rotated = np.mean(x_vals_rotated)
    print(f"\n  Avg X of top players: {avg_top_x_rotated:.1f}")
    print(f"  X difference (top vs bottom): {abs(avg_top_x_rotated - rx14):.1f} pixels")

    # =========================================================================
    # Transform 2: Field Axis Projection (new implementation)
    # =========================================================================
    print("\n" + "="*70)
    print("TRANSFORM 2: Field Axis Projection (proposed fix)")
    print("="*70)

    field_transform = FieldAxisProjectionTransform(
        yard_line_angle=yard_line_angle,
        center_x=center_x,
        center_y=center_y
    )

    print("\nAfter field axis projection:")
    for p in same_yardline_players:
        tx, ty = field_transform.transform_point(p['x'], p['y'])
        print(f"  ID {p['id']:2d}: x={tx:7.1f}, y={ty:7.1f}")

    x_vals_projected = [field_transform.transform_point(p['x'], p['y'])[0] for p in same_yardline_players]
    print(f"  X spread: {max(x_vals_projected)-min(x_vals_projected):.1f} pixels")

    tx14, ty14 = field_transform.transform_point(diff_yardline_player['x'], diff_yardline_player['y'])
    print(f"\n  ID {diff_yardline_player['id']:2d}: x={tx14:7.1f}, y={ty14:7.1f}")

    # Check horizontal separation
    avg_top_x_projected = np.mean(x_vals_projected)
    print(f"\n  Avg X of top players: {avg_top_x_projected:.1f}")
    print(f"  X difference (top vs bottom): {abs(avg_top_x_projected - tx14):.1f} pixels")

    # =========================================================================
    # Summary and Visualization
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    rotated_x_spread = max(x_vals_rotated) - min(x_vals_rotated)
    projected_x_spread = max(x_vals_projected) - min(x_vals_projected)

    print(f"\nX spread among top players (should be MINIMAL for same yardline):")
    print(f"  Basic Rotation: {rotated_x_spread:.1f} pixels (~{rotated_x_spread/19:.1f} yards)")
    print(f"  Field Axis Projection: {projected_x_spread:.1f} pixels (~{projected_x_spread/19:.1f} yards)")

    if projected_x_spread < rotated_x_spread:
        improvement = (rotated_x_spread - projected_x_spread) / rotated_x_spread * 100
        print(f"\n  ✅ Field Axis Projection is {improvement:.0f}% better at preserving spatial relationships!")
    else:
        print(f"\n  ⚠️ Both transforms have similar X spread")

    # Create visualization
    create_comparison_plot(
        same_yardline_players,
        diff_yardline_player,
        rotate_point,
        field_transform.transform_point,
        yard_line_angle
    )

    return {
        'rotated_x_spread': rotated_x_spread,
        'projected_x_spread': projected_x_spread,
        'yard_line_angle': yard_line_angle
    }


def create_comparison_plot(
    top_players,
    bottom_player,
    rotate_fn,
    project_fn,
    yard_line_angle
):
    """Create a visualization comparing the two transforms."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    center_x, center_y = 960, 492

    # Plot 1: Original positions
    ax = axes[0]
    ax.set_title('Original Video Coordinates\n(Notice slanted yardlines)', fontsize=12)

    # Draw slanted yard lines (representative)
    for y_offset in [200, 350, 500, 650, 800]:
        angle_rad = np.radians(yard_line_angle)
        x_start = 200
        x_end = 1800
        y_start = y_offset + (x_start - center_x) * np.tan(angle_rad)
        y_end = y_offset + (x_end - center_x) * np.tan(angle_rad)
        ax.plot([x_start, x_end], [y_start, y_end], 'g-', alpha=0.3, linewidth=1)

    for p in top_players:
        ax.plot(p['x'], p['y'], 'ro', markersize=12, markeredgecolor='black')
        ax.annotate(f'ID {p["id"]}', (p['x'], p['y']), textcoords='offset points',
                   xytext=(8, 8), fontsize=10)

    ax.plot(bottom_player['x'], bottom_player['y'], 'bo', markersize=12, markeredgecolor='black')
    ax.annotate(f'ID {bottom_player["id"]}', (bottom_player['x'], bottom_player['y']),
                textcoords='offset points', xytext=(8, 8), fontsize=10)

    ax.set_xlim(0, 1920)
    ax.set_ylim(984, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)

    # Plot 2: After basic rotation
    ax = axes[1]
    ax.set_title('After Basic Rotation\n(Still has perspective distortion)', fontsize=12)

    # Draw horizontal yard lines
    for y_offset in [200, 350, 500, 650, 800]:
        ax.axhline(y=y_offset, color='g', alpha=0.3, linewidth=1)

    for p in top_players:
        rx, ry = rotate_fn(p['x'], p['y'])
        ax.plot(rx, ry, 'ro', markersize=12, markeredgecolor='black')
        ax.annotate(f'ID {p["id"]}', (rx, ry), textcoords='offset points',
                   xytext=(8, 8), fontsize=10)

    rx14, ry14 = rotate_fn(bottom_player['x'], bottom_player['y'])
    ax.plot(rx14, ry14, 'bo', markersize=12, markeredgecolor='black')
    ax.annotate(f'ID {bottom_player["id"]}', (rx14, ry14),
                textcoords='offset points', xytext=(8, 8), fontsize=10)

    ax.set_xlim(0, 1920)
    ax.set_ylim(984, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)

    # Plot 3: After field axis projection
    ax = axes[2]
    ax.set_title('After Field Axis Projection\n(Correctly preserves spatial relationships)', fontsize=12)

    # Draw horizontal yard lines
    for y_offset in [200, 350, 500, 650, 800]:
        ax.axhline(y=y_offset, color='g', alpha=0.3, linewidth=1)

    x_vals = []
    for p in top_players:
        tx, ty = project_fn(p['x'], p['y'])
        ax.plot(tx, ty, 'ro', markersize=12, markeredgecolor='black')
        ax.annotate(f'ID {p["id"]}', (tx, ty), textcoords='offset points',
                   xytext=(8, 8), fontsize=10)
        x_vals.append(tx)

    tx14, ty14 = project_fn(bottom_player['x'], bottom_player['y'])
    ax.plot(tx14, ty14, 'bo', markersize=12, markeredgecolor='black')
    ax.annotate(f'ID {bottom_player["id"]}', (tx14, ty14),
                textcoords='offset points', xytext=(8, 8), fontsize=10)

    # Draw line showing X positions are now similar for top players
    avg_x = np.mean(x_vals)
    y_min, y_max = ax.get_ylim()
    ax.axvline(x=avg_x, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.annotate(f'Avg X = {avg_x:.1f}', (avg_x, 100), fontsize=10, color='red',
                ha='center')

    ax.set_xlim(0, 1920)
    ax.set_ylim(984, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path("transformation_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison plot to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare transformation methods")
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Path to video (optional, for auto-detecting angle)')

    args = parser.parse_args()

    results = compare_transforms(args.video)

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

if __name__ == "__main__":
    main()