"""
NFL Route Tracker - Visualization Module
=========================================

This module provides visualization tools for trajectories and tracking results.
Good visualizations are essential for:
1. Debugging - seeing if the tracker is working correctly
2. Understanding - intuitively grasping trajectory patterns
3. Presentation - creating professional outputs

Author: MiniMax Agent
Phase: 1 - Foundation

Visualization Types:
-------------------
1. Trajectory plots - 2D paths showing object movement
2. Animated playback - Frame-by-frame with overlaid tracking
3. Heat maps - Where objects spend the most time
4. Speed profiles - How velocity changes over time
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.tracking.trajectory import Trajectory, TrajectoryStore, Detection


class TrajectoryVisualizer:
    """
    Creates visualizations of tracking results.

    This class provides multiple ways to visualize trajectories:
    - Static 2D plots
    - Animated videos
    - Statistical plots

    Example Usage:
    -------------
    ```python
    # Create visualizer
    viz = TrajectoryVisualizer(figsize=(10, 8))

    # Plot all trajectories
    viz.plot_trajectories(trajectory_store, output_path="trajectories.png")

    # Plot with speed coloring
    viz.plot_trajectories_with_speed(trajectory_store, fps=30.0)

    # Create comparison plot
    viz.plot_trajectory_comparison([traj1, traj2], labels=["WR1", "WR2"])
    ```
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        style: str = 'dark_background'
    ):
        """
        Initialize the visualizer.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size (width, height) in inches
        style : str
            Matplotlib style ('dark_background' looks like a playbook)
        """
        self.figsize = figsize
        self.style = style

        # Color palette for multiple trajectories
        self.colors = [
            '#00ff00',  # Green
            '#ff0000',  # Red
            '#0088ff',  # Blue
            '#ffff00',  # Yellow
            '#ff00ff',  # Magenta
            '#00ffff',  # Cyan
            '#ff8800',  # Orange
            '#88ff00',  # Lime
        ]

        print(f"[TrajectoryVisualizer] Initialized")
        print(f"                       Figure size: {figsize}")
        print(f"                       Style: {style}")

    def plot_trajectories(
        self,
        store: TrajectoryStore,
        output_path: Optional[str] = None,
        show: bool = True,
        title: str = "Tracked Trajectories",
        flip_y: bool = True
    ) -> plt.Figure:
        """
        Plot all trajectories from a TrajectoryStore.

        This creates a 2D plot showing the path each tracked object took.

        Parameters:
        -----------
        store : TrajectoryStore
            The tracking results to visualize
        output_path : str, optional
            If provided, save figure to this path
        show : bool
            If True, display the figure
        title : str
            Plot title
        flip_y : bool
            If True, flip Y axis (image coords have Y increasing downward)

        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        print(f"[TrajectoryVisualizer] Creating trajectory plot...")

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            trajectories = store.get_all_trajectories()

            if not trajectories:
                print("                       WARNING: No trajectories to plot!")
                ax.text(0.5, 0.5, "No trajectories", ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
            else:
                for i, traj in enumerate(trajectories):
                    color = self.colors[i % len(self.colors)]
                    self._plot_single_trajectory(ax, traj, color)

            # Formatting
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')

            if flip_y:
                ax.invert_yaxis()  # Match image coordinate system

            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # Add legend
            if trajectories:
                ax.legend(loc='upper right')

            plt.tight_layout()

            # Save if requested
            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"                       Saved to: {output_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()

            return fig

    def _plot_single_trajectory(
        self,
        ax: plt.Axes,
        trajectory: Trajectory,
        color: str,
        linewidth: float = 2.0,
        marker_size: float = 8.0
    ) -> None:
        """
        Plot a single trajectory on the given axes.

        This helper method handles the actual drawing of one trajectory,
        including:
        - The path line
        - Start marker (circle)
        - End marker (square)
        - Direction arrow

        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axes to draw on
        trajectory : Trajectory
            The trajectory to plot
        color : str
            Color for this trajectory
        linewidth : float
            Width of the path line
        marker_size : float
            Size of start/end markers
        """
        frames, xs, ys = trajectory.get_path()

        if len(xs) < 2:
            return

        # Plot the path
        ax.plot(
            xs, ys,
            color=color,
            linewidth=linewidth,
            label=f'Track {trajectory.track_id}',
            alpha=0.8
        )

        # Mark start point (circle)
        ax.scatter(
            xs[0], ys[0],
            s=marker_size**2,
            color=color,
            marker='o',
            edgecolors='white',
            linewidths=1.5,
            zorder=5
        )

        # Mark end point (square)
        ax.scatter(
            xs[-1], ys[-1],
            s=marker_size**2,
            color=color,
            marker='s',
            edgecolors='white',
            linewidths=1.5,
            zorder=5
        )

        # Add direction arrow at midpoint
        mid_idx = len(xs) // 2
        if mid_idx > 0 and mid_idx < len(xs) - 1:
            dx = xs[mid_idx + 1] - xs[mid_idx - 1]
            dy = ys[mid_idx + 1] - ys[mid_idx - 1]

            # Normalize and scale arrow
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx / length * 10
                dy = dy / length * 10

                ax.annotate(
                    '',
                    xy=(xs[mid_idx] + dx, ys[mid_idx] + dy),
                    xytext=(xs[mid_idx], ys[mid_idx]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
                )

    def plot_trajectories_with_speed(
        self,
        store: TrajectoryStore,
        fps: float = 30.0,
        output_path: Optional[str] = None,
        show: bool = True,
        title: str = "Trajectories Colored by Speed"
    ) -> plt.Figure:
        """
        Plot trajectories with color indicating speed.

        This creates a more informative visualization where:
        - The path color changes based on speed
        - Blue = slow, Red = fast
        - Useful for seeing where players accelerate/decelerate

        Parameters:
        -----------
        store : TrajectoryStore
            Tracking results
        fps : float
            Frames per second (needed to calculate speed)
        output_path : str, optional
            Save path
        show : bool
            Display figure
        title : str
            Plot title

        Returns:
        --------
        plt.Figure
        """
        print(f"[TrajectoryVisualizer] Creating speed-colored plot...")

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            trajectories = store.get_all_trajectories()

            # Find global speed range for consistent coloring
            all_speeds = []
            for traj in trajectories:
                speeds = traj.get_speeds(fps)
                if len(speeds) > 0:
                    all_speeds.extend(speeds.tolist())

            if all_speeds:
                vmin, vmax = min(all_speeds), max(all_speeds)
            else:
                vmin, vmax = 0, 1

            # Create colormap
            cmap = plt.cm.coolwarm  # Blue (slow) to Red (fast)
            norm = Normalize(vmin=vmin, vmax=vmax)

            for traj in trajectories:
                self._plot_trajectory_with_colormap(ax, traj, fps, cmap, norm)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Speed (pixels/second)')

            # Formatting
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"                       Saved to: {output_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig

    def _plot_trajectory_with_colormap(
        self,
        ax: plt.Axes,
        trajectory: Trajectory,
        fps: float,
        cmap,
        norm
    ) -> None:
        """Plot trajectory with speed-based coloring using LineCollection."""
        frames, xs, ys = trajectory.get_path()
        speeds = trajectory.get_speeds(fps)

        if len(speeds) < 1:
            return

        # Create line segments
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create colored line collection
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speeds)
        lc.set_linewidth(3)

        ax.add_collection(lc)

    def plot_speed_profile(
        self,
        trajectory: Trajectory,
        fps: float = 30.0,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot speed over time for a single trajectory.

        This creates a line plot showing how speed changes throughout
        the trajectory. Useful for analyzing:
        - When players accelerate (route breaks)
        - When players decelerate (catching ball)
        - Overall route dynamics

        Parameters:
        -----------
        trajectory : Trajectory
            Single trajectory to analyze
        fps : float
            Frames per second
        output_path : str, optional
            Save path
        show : bool
            Display figure

        Returns:
        --------
        plt.Figure
        """
        print(f"[TrajectoryVisualizer] Creating speed profile for Track {trajectory.track_id}...")

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(10, 4))

            frames, _, _ = trajectory.get_path()
            speeds = trajectory.get_speeds(fps)

            if len(speeds) > 0:
                # Time axis (skip first frame since speed is between frames)
                times = frames[1:] / fps

                ax.plot(times, speeds, color='#00ff00', linewidth=2)
                ax.fill_between(times, 0, speeds, alpha=0.3, color='#00ff00')

                # Statistics
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)

                ax.axhline(avg_speed, color='yellow', linestyle='--',
                          label=f'Avg: {avg_speed:.1f} px/s')

                ax.scatter(times[np.argmax(speeds)], max_speed,
                          color='red', s=100, zorder=5,
                          label=f'Max: {max_speed:.1f} px/s')

            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Speed (pixels/second)', fontsize=12)
            ax.set_title(f'Speed Profile - Track {trajectory.track_id}',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"                       Saved to: {output_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig

    def plot_trajectory_comparison(
        self,
        trajectories: List[Trajectory],
        labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        show: bool = True,
        title: str = "Trajectory Comparison"
    ) -> plt.Figure:
        """
        Compare multiple trajectories side by side.

        This is useful for:
        - Comparing routes run by different players
        - Analyzing the same route across different plays
        - Before/after comparisons

        Parameters:
        -----------
        trajectories : List[Trajectory]
            Trajectories to compare
        labels : List[str], optional
            Labels for each trajectory (default: Track IDs)
        output_path : str, optional
            Save path
        show : bool
            Display figure
        title : str
            Plot title

        Returns:
        --------
        plt.Figure
        """
        print(f"[TrajectoryVisualizer] Creating comparison plot...")

        if labels is None:
            labels = [f"Track {t.track_id}" for t in trajectories]

        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, len(trajectories),
                                      figsize=(5*len(trajectories), 5))

            if len(trajectories) == 1:
                axes = [axes]

            for ax, traj, label, color in zip(axes, trajectories, labels, self.colors):
                frames, xs, ys = traj.get_path()

                if len(xs) > 0:
                    ax.plot(xs, ys, color=color, linewidth=2)
                    ax.scatter(xs[0], ys[0], s=100, color=color, marker='o',
                              edgecolors='white', label='Start')
                    ax.scatter(xs[-1], ys[-1], s=100, color=color, marker='s',
                              edgecolors='white', label='End')

                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')

                # Add stats
                stats_text = (
                    f"Detections: {len(traj)}\n"
                    f"Distance: {traj.get_total_distance():.0f}px\n"
                    f"Displacement: {traj.get_displacement():.0f}px"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"                       Saved to: {output_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig


def create_tracking_video(
    video_path: str,
    store: TrajectoryStore,
    output_path: str,
    fps: Optional[float] = None
) -> None:
    """
    Create a video with tracking overlays.

    This renders the original video with bounding boxes and
    trajectory trails drawn on each frame.

    Parameters:
    -----------
    video_path : str
        Path to original video
    store : TrajectoryStore
        Tracking results
    output_path : str
        Output video path
    fps : float, optional
        Output FPS (default: same as input)
    """
    from nfl_route_tracker.core.video_loader import VideoLoader

    print(f"[create_tracking_video] Creating annotated video...")

    with VideoLoader(video_path) as video:
        if fps is None:
            fps = video.metadata.fps

        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (video.metadata.width, video.metadata.height)
        )

        # Build frame->detections lookup
        frame_detections = {}
        for traj in store.get_all_trajectories():
            for det in traj.detections:
                if det.frame_id not in frame_detections:
                    frame_detections[det.frame_id] = []
                frame_detections[det.frame_id].append((traj.track_id, det))

        # Process frames
        for frame_id, frame in video:
            # Draw current detections
            if frame_id in frame_detections:
                for track_id, det in frame_detections[frame_id]:
                    # Draw box
                    x, y = int(det.x), int(det.y)
                    w, h = int(det.width), int(det.height)
                    color = (0, 255, 0)  # Green

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"ID:{track_id}",
                               (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, color, 2)

                    # Draw center
                    cx, cy = det.center
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            writer.write(frame)

        writer.release()

    print(f"[create_tracking_video] Saved to: {output_path}")

if __name__ == "__main__":
    from pathlib import Path
    from nfl_route_tracker.tracking.motion_tracker import MotionTracker
    from nfl_route_tracker.core.config import MotionTrackerConfig

    print("\n" + "="*60)
    print("Testing Visualization Module")
    print("="*60 + "\n")

    # Run motion tracker to get real data
    test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
    video_path = test_folder / "trial_vid.mp4"
    output_folder = test_folder.parent / "viz_output"
    output_folder.mkdir(exist_ok=True)

    print("Running motion tracker to get trajectory data...")
    config = MotionTrackerConfig()
    tracker = MotionTracker(config)
    store = tracker.process_video(str(video_path))
    print(f"Got {store.num_trajectories} trajectories\n")

    viz = TrajectoryVisualizer(figsize=(10, 8), style='dark_background')

    # Test 1: Basic trajectory plot
    print("="*60)
    print("TEST 1: Basic trajectory plot")
    print("="*60)
    fig = viz.plot_trajectories(
        store,
        output_path=str(output_folder / "trajectories.png"),
        show=False,
        title="Test Linear - Tracked Trajectories"
    )
    assert fig is not None
    print("PASSED!\n")

    # Test 2: Speed-colored plot
    print("="*60)
    print("TEST 2: Speed-colored trajectory plot")
    print("="*60)
    fig = viz.plot_trajectories_with_speed(
        store,
        fps=30.0,
        output_path=str(output_folder / "speed_colored.png"),
        show=False,
        title="Test Linear - Speed Colored"
    )
    assert fig is not None
    print("PASSED!\n")

    # Test 3: Speed profile for first trajectory
    print("="*60)
    print("TEST 3: Speed profile")
    print("="*60)
    traj = store.get_all_trajectories()[0]
    fig = viz.plot_speed_profile(
        traj,
        fps=30.0,
        output_path=str(output_folder / "speed_profile.png"),
        show=False
    )
    assert fig is not None
    print("PASSED!\n")

    # Test 4: Comparison plot across multiple videos
    print("="*60)
    print("TEST 4: Trajectory comparison across videos")
    print("="*60)
    trajectories = []
    labels = []
    for video_file in sorted(test_folder.glob("*.mp4"))[:3]:
        tracker.reset()
        result = tracker.process_video(str(video_file))
        if result.num_trajectories > 0:
            trajectories.append(result.get_all_trajectories()[0])
            labels.append(video_file.stem)

    fig = viz.plot_trajectory_comparison(
        trajectories,
        labels=labels,
        output_path=str(output_folder / "comparison.png"),
        show=False,
        title="Route Comparison Across Test Videos"
    )
    assert fig is not None
    print("PASSED!\n")

    # Test 5: Annotated tracking video
    print("="*60)
    print("TEST 5: Annotated tracking video")
    print("="*60)
    tracker.reset()
    store = tracker.process_video(str(video_path))
    output_video = str(output_folder / "tracked_output.mp4")
    create_tracking_video(str(video_path), store, output_video)
    assert Path(output_video).exists()
    print("PASSED!\n")

    print("="*60)
    print(f"All visualization tests passed!")
    print(f"Outputs saved to: {output_folder}")
    print("="*60)


# if __name__ == "__main__":
#     """
#     Test the visualization module.
#     """
#     print("\n" + "="*60)
#     print("Testing Visualization Module")
#     print("="*60 + "\n")

#     # Create sample data
#     print("[TEST SETUP] Creating sample trajectory data...")

#     store = TrajectoryStore()

#     # Create a trajectory that looks like a receiver route (out route)
#     # Start, run straight, then break right
#     np.random.seed(42)  # Reproducibility

#     for frame in range(60):
#         if frame < 20:
#             # Running straight up
#             x = 160 + np.random.normal(0, 2)
#             y = 200 - frame * 3
#         else:
#             # Break right
#             x = 160 + (frame - 20) * 4 + np.random.normal(0, 2)
#             y = 200 - 60 + np.random.normal(0, 2)

#         det = Detection(
#             frame_id=frame,
#             x=x - 15,
#             y=y - 25,
#             width=30,
#             height=50
#         )
#         store.add_detection(track_id=0, detection=det)

#     # Add second trajectory (curl route)
#     for frame in range(60):
#         if frame < 30:
#             x = 50 + np.random.normal(0, 2)
#             y = 200 - frame * 3
#         else:
#             # Curl back
#             x = 50 + np.random.normal(0, 2)
#             y = 200 - 90 + (frame - 30) * 1.5

#         det = Detection(
#             frame_id=frame,
#             x=x - 15,
#             y=y - 25,
#             width=30,
#             height=50
#         )
#         store.add_detection(track_id=1, detection=det)

#     print(store.get_summary())

#     # Test 1: Basic trajectory plot
#     print("\n[TEST 1] Creating basic trajectory plot...")
#     viz = TrajectoryVisualizer(figsize=(8, 6))
#     viz.plot_trajectories(
#         store,
#         output_path="test_trajectories.png",
#         show=False,
#         title="Test Routes (Out + Curl)"
#     )
#     print("         PASSED!\n")

#     # Test 2: Speed-colored plot
#     print("[TEST 2] Creating speed-colored plot...")
#     viz.plot_trajectories_with_speed(
#         store,
#         fps=30.0,
#         output_path="test_trajectories_speed.png",
#         show=False
#     )
#     print("         PASSED!\n")

#     # Test 3: Speed profile
#     print("[TEST 3] Creating speed profile...")
#     traj = store.get_trajectory(0)
#     viz.plot_speed_profile(
#         traj,
#         fps=30.0,
#         output_path="test_speed_profile.png",
#         show=False
#     )
#     print("         PASSED!\n")

#     # Test 4: Comparison plot
#     print("[TEST 4] Creating comparison plot...")
#     viz.plot_trajectory_comparison(
#         [store.get_trajectory(0), store.get_trajectory(1)],
#         labels=["Out Route", "Curl Route"],
#         output_path="test_comparison.png",
#         show=False
#     )
#     print("         PASSED!\n")

#     # Cleanup
#     for f in ["test_trajectories.png", "test_trajectories_speed.png",
#               "test_speed_profile.png", "test_comparison.png"]:
#         Path(f).unlink(missing_ok=True)

#     print("="*60)
#     print("All Visualization tests passed!")
#     print("="*60)

