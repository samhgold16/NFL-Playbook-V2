"""
NFL Route Tracker - Visualization Module
=========================================

This module provides visualization tools for trajectories and tracking results.
"""

# important packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Import important modules
from nfl_route_tracker.tracking.trajectory import Trajectory, TrajectoryStore
from nfl_route_tracker.core.video_loader import VideoLoader

class TrajectoryVisualizer:
    """
    Creates visualizations of tracking results.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'dark_background'):
        """
        Initialize the visualizer.
        """
        self.figsize = figsize
        self.style = style

        # Color palette for multiple trajectories
        # NEED TO FIX LATER, SHOULD HAVE 22 COLORS, 
        # Way to split into 11x11 and have two separate colorscales for offense/defense?
        # will also have to change how `color` is implemented in following plots
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

    def plot_trajectories(self, store: TrajectoryStore, output_path: Optional[str] = None, title: str = "Tracked Trajectories",
                          flip_y: bool = False, video_height: float = 984,
                          invert_y_axis: bool = True) -> plt.Figure:
        """
        Plot all trajectories from a TrajectoryStore object.

        IMPORTANT - Video Coordinate Convention:
        =========================================
        In video coordinates:
        - Y=0 is at the TOP of the frame
        - Y increases going DOWN toward bottom of frame

        In matplotlib (default):
        - Y=0 is at the BOTTOM of the plot
        - Y increases going UP toward top of plot

        By default, we INVERT the y-axis so that the plot matches the video:
        - Plot TOP = Video TOP (far side of field)
        - Plot BOTTOM = Video BOTTOM (near side of field)

        Args:
            store: TrajectoryStore with trajectory data
            output_path: Where to save the plot
            title: Plot title
            flip_y: DEPRECATED - Use invert_y_axis instead.
                   Kept for backwards compatibility.
            video_height: Video height for proper coordinate interpretation
            invert_y_axis: If True (default), invert y-axis so plot matches video
                          Set to False if you want standard matplotlib coordinates
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize = self.figsize)

            trajectories = store.get_all_trajectories()

            if not trajectories:
                print("WARNING: No trajectories to plot!")
                ax.text(0.5, 0.5, "No trajectories", ha = 'center', va = 'center',
                       transform = ax.transAxes, fontsize = 16)
            else:
                for i, traj in enumerate(trajectories):
                    color = self.colors[i % len(self.colors)]
                    # function for plotting each trace below
                    self._plot_single_trajectory(ax, traj, color)

            # Formatting with CORRECT coordinate semantics
            ax.set_xlabel('X Position (field length, toward endzone)', fontsize = 12)
            ax.set_ylabel('Y Position (field width, sideline to sideline)', fontsize = 12)
            ax.set_title(f"{title}\n(X = up/down field, Y = sideline to sideline)\n"
                        f"[Y-axis {'INVERTED: top=video TOP, bottom=video BOTTOM' if invert_y_axis else 'NOT inverted: standard matplotlib'}]",
                        fontsize = 11, fontweight = 'bold')

            # Video coordinates: Y=0 at TOP, Y increases going DOWN
            # Invert y-axis so plot matches video orientation
            if invert_y_axis:
                ax.invert_yaxis()

            ax.grid(True, alpha = 0.3)
            ax.set_aspect('equal')

            # Add legend, can show when reduced down to 22
            if trajectories:
                ax.legend(loc = 'upper right')

            plt.tight_layout()

            # Save if requested
            if output_path:
                fig.savefig(output_path, dpi = 150, bbox_inches = 'tight')
                print(f"Saved Trajectory plot to: {output_path}")

            return fig

    def _plot_single_trajectory(self, ax: plt.Axes, trajectory: Trajectory,
        color: str, linewidth: float = 2.0, marker_size: float = 8.0) -> None:
        """
        Plot a single trajectory on the given axes.
        Used in the main plot_trajectories
        """
        frames, xs, ys = trajectory.get_path()

        if len(xs) < 2:
            return

        # Plot the path
        ax.plot(xs, ys, color = color, linewidth = linewidth, label = f'Track {trajectory.track_id}', alpha = 0.8)

        # Mark start point (circle)
        ax.scatter(xs[0], ys[0], s = marker_size**2, color = color, marker = 'o',
                   edgecolors = 'white', linewidths = 1.5, zorder = 5)

        # Mark end point (square)
        ax.scatter(xs[-1], ys[-1], s = marker_size**2, color = color, marker = 's',
                   edgecolors = 'white', linewidths = 1.5,zorder = 5)

    def plot_trajectories_with_field_context(self, store: TrajectoryStore,
                                              output_path: Optional[str] = None,
                                              title: str = "Trajectories on Field",
                                              video_width: float = 1920,
                                              video_height: float = 984,
                                              invert_y_axis: bool = True) -> plt.Figure:
        """
        Plot trajectories with field context and proper axis interpretation.

        This method provides a more intuitive visualization by:
        1. Using proper axis labels (X = field length, Y = field width)
        2. Adding route direction interpretation
        3. Classifying primary route direction for each trajectory

        IMPORTANT - Video Coordinate Convention:
        =========================================
        In video coordinates:
        - Y=0 is at the TOP of the frame
        - Y increases going DOWN toward bottom of frame

        By default, we INVERT the y-axis so that the plot matches the video.

        Args:
            store: TrajectoryStore with trajectory data
            output_path: Where to save the plot
            title: Plot title
            video_width: Video width for position normalization
            video_height: Video height for position normalization
            invert_y_axis: If True (default), invert y-axis so plot matches video

        Returns:
            matplotlib Figure
        """
        positions = store.get_all_field_positions(video_width, video_height)

        # Local style for this plot
        linewidth = 2.0
        marker_size = 8.0

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize = self.figsize)

            trajectories = store.get_all_trajectories()

            if not trajectories:
                print("WARNING: No trajectories to plot!")
                ax.text(0.5, 0.5, "No trajectories", ha = 'center', va = 'center',
                       transform = ax.transAxes, fontsize = 16)
            else:
                for i, traj in enumerate(trajectories):
                    color = self.colors[i % len(self.colors)]
                    field_pos = positions.get(traj.track_id, {})

                    frames, xs, ys = traj.get_path()
                    if len(xs) < 2:
                        continue

                    # Determine route type based on movement
                    direction = field_pos.get('primary_direction', 'unknown')
                    route_label = f"Track {traj.track_id}"
                    if direction != 'unknown':
                        direction_symbol = "↕" if direction == 'vertical' else "↔"
                        route_label += f" ({direction_symbol} {direction})"

                    ax.plot(xs, ys, color = color, linewidth = linewidth,
                           label = route_label, alpha = 0.8)
                    ax.scatter(xs[0], ys[0], s = marker_size**2, color = color, marker = 'o',
                              edgecolors = 'white', linewidths = 1.5, zorder = 5)
                    ax.scatter(xs[-1], ys[-1], s = marker_size**2, color = color, marker = 's',
                              edgecolors = 'white', linewidths = 1.5, zorder = 5)

            ax.set_xlabel('X = Field Length (horizontal, toward endzone →)', fontsize = 12)
            ax.set_ylabel('Y = Field Width (vertical, sideline to sideline)', fontsize = 12)
            ax.set_title(f"{title}\n(↕ = vertical route, ↔ = horizontal route)\n"
                        f"[Y-axis {'INVERTED: top=video TOP, bottom=video BOTTOM' if invert_y_axis else 'NOT inverted: standard matplotlib'}]",
                        fontsize = 11, fontweight = 'bold')

            # Video coordinates: Y=0 at TOP, Y increases going DOWN
            # Invert y-axis so plot matches video orientation
            if invert_y_axis:
                ax.invert_yaxis()

            ax.grid(True, alpha = 0.3)
            ax.set_aspect('equal')
            ax.legend(loc = 'upper right', fontsize = 8)

            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi = 150, bbox_inches = 'tight')
                print(f"Saved trajectory plot to: {output_path}")

            return fig


def create_tracking_video(video_path: str, store: TrajectoryStore, output_path: str, fps: Optional[float] = None) -> None:
    """
    Create a video with trackin and bounding boxes overlays.
    """

    print(f"Creating updated tracking video...")

    with VideoLoader(video_path) as video:
        if fps is None:
            fps = video.metadata.fps

        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc,
                                 fps, (video.metadata.width, video.metadata.height))

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
                    color = (180, 50, 60)  

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
                    cv2.putText(frame, f"ID:{track_id}",
                               (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, color, 4)

                    # Draw center
                    cx, cy = det.center
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            writer.write(frame)

        writer.release()

    print(f"Updated video saved to: {output_path}")