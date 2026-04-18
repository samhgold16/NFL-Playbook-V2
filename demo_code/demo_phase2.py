#!/usr/bin/env python3
"""NFL Route Tracker - Phase 2 Demo
===================================
Demonstrates the Phase 2 workflow for skill position player identification.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import Phase 2 modules
from nfl_route_tracker.skill_players import (
    TrajectoryDataLoader,
    load_trajectory_from_json,
    LineOfScrimmageClassifier,
    classify_offense_defense,
    SkillPositionFilter,
    filter_skill_position_players,
    PlayerClassification,
    #SyntheticRouteGenerator,
    #generate_synthetic_dataset,
    #RouteType,
    #ALL_ROUTE_TYPES,
)
from nfl_route_tracker.tracking.trajectory import TrajectoryStore
from nfl_route_tracker.core.video_loader import VideoLoader


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print("\n" + "-" * 50)
    print(f"  {text}")
    print("-" * 50)


class Phase2Visualizer:
    """
    Visualizer for Phase 2 results.
    """
 
    # Color scheme
    COLORS = {
        'offense_skill':     '#ffffff',   # WHITE  — skill position offense
        'offense_non_skill': '#00cc00',   # GREEN  — non-skill offense (OL/QB)
        'defense':           '#ff3333',   # RED    — defense
        'los':               '#ffff00',   # YELLOW — LOS line (distinct from white trajectories)
    }
 
    # Marker scheme
    MARKERS = {
        'skill':     'o',   # Circle for skill position
        'non_skill': 's',   # Square for non-skill
        'defense':   '^',   # Triangle for defense
    }
 
    def __init__(self, figsize: tuple = (14, 8)):
        self.figsize = figsize
 
    def _draw_los(self, ax, los, video_height: float) -> None:
        """
        Draw the LOS — handles both vertical and sloped lines correctly
        by sampling get_x_at_y() across the full frame height.
        """
        ys = np.linspace(0, video_height, 100)
        xs = np.array([los.get_x_at_y(y) for y in ys])
        ax.plot(xs, ys, color=self.COLORS['los'], linestyle='--',
                linewidth=2, label='Line of Scrimmage', alpha=0.8)
 
    def _plot_category(self, ax, trajectories, color, marker, label) -> None:
        """Plot a category of trajectories."""
        for i, traj in enumerate(trajectories):
            frames, xs, ys = traj.get_center_coords()
            if len(xs) < 2:
                continue
            # Only attach the label to the first trajectory so the legend
            # doesn't repeat the same entry for every player
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7,
                    label=label if i == 0 else None)
            ax.scatter(xs[0], ys[0], s=80, color=color, marker='o',
                       edgecolors='black', linewidths=1.0, zorder=5)
            ax.scatter(xs[-1], ys[-1], s=80, color=color, marker=marker,
                       edgecolors='black', linewidths=1.0, zorder=5)
            
    def plot_offense_defense_only(self, trajectories, los, video_width,
                                  video_height, title="Offense vs Defense") -> plt.Figure:
        """Plot offense/defense split only (no skill position labels)."""
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=self.figsize)
 
            offense = [t for t in trajectories if t.team == 'offense']
            defense = [t for t in trajectories if t.team == 'defense']
 
            self._plot_category(ax, offense, self.COLORS['offense_non_skill'],
                                self.MARKERS['non_skill'], 'Offense')
            self._plot_category(ax, defense, self.COLORS['defense'],
                                self.MARKERS['defense'], 'Defense')
            self._draw_los(ax, los, video_height)
 
            ax.set_xlim(-50, video_width + 50)
            ax.set_ylim(-50, video_height + 50)
            ax.set_xlabel('X (Field Length →)', fontsize=11)
            ax.set_ylabel('Y (Field Width ↑)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.tight_layout()
            return fig

    def plot_phase2_results(self,
                            trajectories: List,
                            los,                        # LineOfScrimmage object
                            video_width: float,
                            video_height: float,
                            output_path: Optional[str] = None,
                            title: str = "Phase 2: Offense/Defense and Skill Position Classification"
                            ) -> plt.Figure:
        """
        Plot trajectories colored by offense/defense and skill position.
        """
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=self.figsize)
 
            # Split into the three display categories
            offense_skill     = [t for t in trajectories
                                  if t.team == 'offense' and t.is_skill_position]
            offense_non_skill = [t for t in trajectories
                                  if t.team == 'offense' and not t.is_skill_position]
            defense           = [t for t in trajectories if t.team == 'defense']
 
            # Plot each category
            self._plot_category(ax, offense_skill,     self.COLORS['offense_skill'],
                                self.MARKERS['skill'],     'Offense — Skill (WR/TE/RB)')
            self._plot_category(ax, offense_non_skill,  self.COLORS['offense_non_skill'],
                                self.MARKERS['non_skill'],  'Offense — Non-Skill (OL/QB)')
            self._plot_category(ax, defense,            self.COLORS['defense'],
                                self.MARKERS['defense'],    'Defense')
 
            # Draw LOS using the object directly (handles slope correctly)
            self._draw_los(ax, los, video_height)
 
            # Annotate LOS — sample x at mid-frame height for label position
            los_x_mid = los.get_x_at_y(video_height / 2)
            ax.text(los_x_mid + 20, video_height * 0.05, 'LOS', rotation=90,
                    color=self.COLORS['los'], fontsize=10, fontweight='bold',
                    ha='left', va='bottom')
 
            # Axis formatting
            ax.set_xlabel('X Position (Field Length →)', fontsize=12)
            ax.set_ylabel('Y Position (Field Width ↑)', fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlim(-50, video_width + 50)
            ax.set_ylim(-50, video_height + 50)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
 
            # Manual legend so each category appears exactly once
            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.COLORS['offense_skill'],
                       markersize=10, label='Offense — Skill Position (WR/TE/RB)'),
                Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=self.COLORS['offense_non_skill'],
                       markersize=10, label='Offense — Non-Skill (OL/QB)'),
                Line2D([0], [0], marker='^', color='w',
                       markerfacecolor=self.COLORS['defense'],
                       markersize=10, label='Defense'),
                Line2D([0], [0], color=self.COLORS['los'], linestyle='--',
                       linewidth=2, label='Line of Scrimmage'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
 
            plt.tight_layout()
 
            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  Saved plot to: {output_path}")
 
            return fig

    def plot_synthetic_routes(self,
                              synthetic_routes: List,
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot synthetic route examples in normalized coordinates.

        Args:
            synthetic_routes: List of SyntheticRoute objects
            output_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(10, 10))

            # Group by route type
            routes_by_type = {}
            for route in synthetic_routes:
                if route.route_type.value not in routes_by_type:
                    routes_by_type[route.route_type.value] = []
                routes_by_type[route.route_type.value].append(route)

            # Color map for routes
            colors = plt.cm.tab20(np.linspace(0, 1, len(routes_by_type)))
            color_map = {rt: colors[i] for i, rt in enumerate(routes_by_type.keys())}

            # Plot a few examples of each type
            max_per_type = 5
            for route_type, routes in routes_by_type.items():
                for route in routes[:max_per_type]:
                    ax.plot(route.x_coords, route.y_coords,
                           color=color_map[route_type], linewidth=2, alpha=0.7)

                # Add label at mean position
                if routes:
                    mean_x = np.mean(routes[0].x_coords)
                    mean_y = np.mean(routes[0].y_coords)
                    ax.text(mean_x, mean_y, route_type.upper(),
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel('X (Field Length - upfield ->)', fontsize=12)
            ax.set_ylabel('Y (Field Width - sideline to sideline)', fontsize=12)
            ax.set_title('Synthetic Route Examples\n'
                        '[In normalized coords: X = upfield, Y = lateral]',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.invert_yaxis()

            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved synthetic routes plot to: {output_path}")

            return fig

def get_video_dimensions(video_path: Path) -> Tuple[float, float]:
    """
    Open a video file briefly just to extract width and height, then close it.
    Returns (width, height) in pixels.
    """
    with VideoLoader(video_path) as vl:
        width = vl.metadata.width
        height = vl.metadata.height
    return float(width), float(height)

SHOW_DUAL_PLOTS = True
def process_single_json(json_path: Path,
                        output_dir: Path,
                        video_width: float = 1920,
                        video_height: float = 984) -> dict:
    """
    Process a single trajectory JSON file through Phase 2.
    """
    print(f"\nProcessing: {json_path.name}")
 
    # Load trajectories
    loader = TrajectoryDataLoader()
    video_trajectories = loader.load_trajectory_json(json_path)
    print(f"  Loaded {video_trajectories.num_trajectories} trajectories")
 
    # ------------------------------------------------------------------
    # Step 1: Offense / Defense Classification
    # ------------------------------------------------------------------
    print_section("Step 1: Offense/Defense Classification")
    trajectories = video_trajectories.trajectories
 
    trajectories, los = classify_offense_defense(
        trajectories,
        method='minimal_movement',
        video_width=video_width,
        video_height=video_height
    )
 
    # get_x_at_y(mid) gives a representative scalar for logging/summary
    los_x_mid = los.get_x_at_y(video_height / 2)
    print(f"  Line of Scrimmage at X ≈ {los_x_mid:.1f} (confidence: {los.confidence:.2f})")
    print("  (LOS is VERTICAL — separates teams along field length)")
 
    offense_count = sum(1 for t in trajectories if t.team == 'offense')
    defense_count = sum(1 for t in trajectories if t.team == 'defense')
    print(f"  Offense players: {offense_count}")
    print(f"  Defense players: {defense_count}")
 
    # TESTING: dual plot — Step 1 result before skill filtering
    if SHOW_DUAL_PLOTS:
        visualizer = Phase2Visualizer(figsize=(14, 8))
        fig1 = visualizer.plot_offense_defense_only(
            trajectories=trajectories,
            los=los,
            video_width=video_width,
            video_height=video_height,
            title=f"Step 1 — Offense vs Defense: {json_path.stem}"
        )
        plot1_path = output_dir / f"{json_path.stem}_step1_offense_defense.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        print(f"  Saved step 1 plot: {plot1_path}")
        plt.close(fig1)
 
    # ------------------------------------------------------------------
    # Step 2: Skill Position Filtering
    # ------------------------------------------------------------------
    print_section("Step 2: Skill Position Filtering")
 
    skill_filter = SkillPositionFilter(los, los_buffer=200.0)
    skill_positions, non_skill = skill_filter.filter_skill_positions(trajectories)
 
    print(f"  Skill position players: {len(skill_positions)}")
    print(f"  Non-skill players: {len(non_skill)}")
 
    position_counts = {}
    for t in trajectories:
        pos = t.position or 'UNKNOWN'
        position_counts[pos] = position_counts.get(pos, 0) + 1
 
    print("\n  Position breakdown:")
    for pos, count in sorted(position_counts.items()):
        is_skill = any(t.position == pos and t.is_skill_position
                       for t in trajectories if t.position == pos)
        print(f"    {pos}: {count} {'(SKILL)' if is_skill else ''}")
 
    # ------------------------------------------------------------------
    # Step 3: Final visualization
    # ------------------------------------------------------------------
    print_section("Step 3: Visualization")
 
    visualizer = Phase2Visualizer(figsize=(14, 8))
    output_path = output_dir / f"{json_path.stem}_phase2.png"
 
    fig2 = visualizer.plot_phase2_results(
        trajectories=trajectories,
        los=los,                        # pass the LineOfScrimmage object
        video_width=video_width,
        video_height=video_height,
        output_path=str(output_path),
        title=f"Phase 2 Classification: {json_path.stem}"
    )
    plt.close(fig2)
 
    return {
        'video_name': json_path.stem,
        'total_trajectories': len(trajectories),
        'offense_count': offense_count,
        'defense_count': defense_count,
        'skill_position_count': len(skill_positions),
        'los_x': los_x_mid,
        'position_breakdown': position_counts,
        'output_path': str(output_path)
    }


# def demo_synthetic_routes(output_dir: Path) -> None:
#     """
#     Generate and display synthetic route examples.

#     Args:
#         output_dir: Directory to save outputs
#     """
#     print_section("Synthetic Route Generation Demo")

#     # Generate dataset
#     generator = SyntheticRouteGenerator(random_seed=42)
#     synthetic_routes = generator.generate_dataset(
#         route_types=ALL_ROUTE_TYPES,
#         routes_per_type=10,  # Few examples for visualization
#         param_variations=True
#     )

#     print(f"  Generated {len(synthetic_routes)} synthetic routes")

#     # Visualize
#     visualizer = Phase2Visualizer()
#     output_path = output_dir / "synthetic_routes_demo.png"

#     fig = visualizer.plot_synthetic_routes(synthetic_routes, str(output_path))
#     plt.close(fig)

#     print(f"  Saved synthetic routes visualization to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Route Tracker Phase 2 Demo - Skill Position Identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Camera Coordinate System (All-22 View):
  X-axis = Field LENGTH (players run up/down field = horizontal in video)
  Y-axis = Field WIDTH (sideline to sideline = vertical in video)
  LOS = VERTICAL line at X position

Route Movement Patterns:
  Go/Streak: Horizontal movement (X changes, Y constant)
  Drag: Vertical movement (Y changes, X increases)
        """
    )
    parser.add_argument('--json', type=str, default=None,
                       help='Path to trajectory JSON file (optional)')
    parser.add_argument('--dir', type=str, default=None,
                       help='Directory containing trajectory JSON files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: data/viz_output2/)')
    parser.add_argument('--video-width', type=float, default=1920,
                       help='Video width (default: 1920)')
    parser.add_argument('--video-height', type=float, default=984,
                       help='Video height (default: 984)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic route examples')

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    default_output_dir = project_root / "data" / "viz_output2"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("NFL Route Tracker - Phase 2 Demo")
    print("Skill Position Player Identification")
    print("-" * 50)

    # Process single JSON file
    if args.json:
        json_path = Path(args.json)
        if not json_path.is_absolute():
            json_path = project_root / args.json
 
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}")
            sys.exit(1)
 
        mp4_files = list(json_path.parent.glob('*.mp4'))
        if mp4_files:
            video_width, video_height = get_video_dimensions(mp4_files[0])
            print(f"Detected video dimensions: {int(video_width)} x {int(video_height)}")
        else:
            print("WARNING: No companion .mp4 found — using --video-width/--video-height defaults.")
            video_width = args.video_width
            video_height = args.video_height
 
        result = process_single_json(
            json_path, output_dir,
            video_width=video_width,
            video_height=video_height
        )

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Video: {result['video_name']}")
        print(f"  Total Players: {result['total_trajectories']}")
        print(f"  Offense: {result['offense_count']}")
        print(f"  Defense: {result['defense_count']}")
        print(f"  Skill Positions: {result['skill_position_count']}")
        print(f"  LOS X-position: {result['los_x']:.1f}")

    # Process directory
    elif args.dir:
        json_dir = Path(args.dir)
        if not json_dir.is_absolute():
            json_dir = project_root / args.dir
 
        if not json_dir.exists():
            print(f"ERROR: Directory not found: {json_dir}")
            sys.exit(1)
 
        json_files = sorted(json_dir.glob('*_trajectories.json'))
        print(f"\nFound {len(json_files)} trajectory files")
 
        results = []
        for json_path in json_files:
            try:
                result = process_single_json(
                    json_path, output_dir,
                    video_width=args.video_width,
                    video_height=args.video_height
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR processing {json_path.name}: {e}")
 
        print("\n" + "=" * 50)
        print("BATCH SUMMARY")
        print("=" * 50)
        print(f"  Processed: {len(results)}/{len(json_files)} files")
        total_skill = sum(r['skill_position_count'] for r in results)
        print(f"  Total skill position trajectories: {total_skill}")
        if results:
            print(f"  Average skill positions per play: {total_skill / len(results):.1f}")

    # Generate synthetic routes demo
    # if args.synthetic:
    #     demo_synthetic_routes(output_dir)

    # Default: Find first JSON in viz_output2
    if not args.json and not args.dir and not args.synthetic:
        # Look for JSON files in viz_output subdirectories
        viz_output = project_root / "data" / "viz_output"

        if viz_output.exists():
            # Find first subdirectory with trajectory JSON
            for subdir in sorted(viz_output.iterdir()):
                if not subdir.is_dir():
                    continue
                json_files = list(subdir.glob('*_trajectories.json'))
                if not json_files:
                    continue

                # Find the companion .mp4 in the same folder
                mp4_files = list(subdir.glob('*.mp4'))
                if not mp4_files:
                    print(f"  WARNING: No .mp4 found in {subdir}, skipping.")
                    continue

                json_path = json_files[0]
                mp4_path = mp4_files[0]
                print(f"\nAuto-selected JSON: {json_path.name}")
                print(f"Companion video:    {mp4_path.name}")

                # Pull true dimensions directly from the video file
                video_width, video_height = get_video_dimensions(mp4_path)
                print(f"Video dimensions:   {int(video_width)} x {int(video_height)}")

                result = process_single_json(
                    json_files[0], output_dir,
                    video_width=video_width,
                    video_height=video_height
                )

                print("\n" + "=" * 50)
                print("SUMMARY")
                print("=" * 50)
                print(f"  Video: {result['video_name']}")
                print(f"  Total Players: {result['total_trajectories']}")
                print(f"  Offense: {result['offense_count']}")
                print(f"  Defense: {result['defense_count']}")
                print(f"  Skill Positions: {result['skill_position_count']}")
                break
            else:
                print("\nNo trajectory JSON files found in data/viz_output/")
                print("\nGenerating synthetic route demo instead...")
                #demo_synthetic_routes(output_dir)
        else:
            print("\nNo data/viz_output/ directory found.")
            print("Generating synthetic route demo instead...")
            #demo_synthetic_routes(output_dir)

    print("\n" + "=" * 70)
    print("  Phase 2 Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
