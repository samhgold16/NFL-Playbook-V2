"""NFL Route Tracker - Phase 2: Offense/Defense Classification
============================================================
Classifies players as offensive or defensive using the Line of Scrimmage.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from .data_loader import ParsedTrajectory, VideoTrajectories


@dataclass
class LineOfScrimmage:
    """
    Represents the Line of Scrimmage as a dividing line.
    """
    # Line parameters (x = my + b, but typically m ≈ 0 for All-22)
    slope: float = 0.0
    intercept: float = 0.0

    # For vertical line approximation (standard in All-22)
    x_value: Optional[float] = None  # If line is approximately vertical

    # Points defining the line segment
    point1: Tuple[float, float] = (0.0, 0.0)
    point2: Tuple[float, float] = (0.0, 0.0)

    confidence: float = 0.0

    @classmethod
    def from_vertical_line(cls, x_value: float, video_height: float,
                            confidence: float = 1.0) -> 'LineOfScrimmage':
        """
        Create a vertical LOS from an x-value.
        """
        return cls(slope=0.0, intercept=x_value, x_value=x_value, point1=(x_value, 0), point2=(x_value, video_height), confidence=confidence)

    @classmethod
    def from_sloped_line(cls, slope: float, intercept: float,
                         video_height: float, confidence: float = 1.0) -> 'LineOfScrimmage':
        """
        Create a sloped LOS from slope and intercept.
        """
        return cls(slope=slope, intercept=intercept,  x_value=None,  # Not a pure vertical line
                    point1=(intercept, 0), point2=(slope * video_height + intercept, video_height),
                    confidence=confidence)

    def get_x_at_y(self, y: float) -> float:
        """Get the x-value of the LOS at a given y position."""
        if self.x_value is not None:
            return self.x_value
        return self.slope * y + self.intercept

    def classify_position(self, x: float, y: float) -> str:
        """
        Classify a point as 'offense' or 'defense' based on LOS position.
        """
        los_x = self.get_x_at_y(y)
        return 'offense' if x < los_x else 'defense'


def fit_line_of_scrimmage(trajectories: List[ParsedTrajectory],
                          method: str = 'clustering',
                          video_width: float = 1920,
                          video_height: float = 1080,
                          first_n_frames: int = 15) -> 'LineOfScrimmage':
    """
    Fit a Line of Scrimmage to trajectories.
    """
    first_positions = []
    for traj in trajectories:
        first_pos = traj.get_first_position()
        if first_pos is not None:
            first_positions.append((traj.track_id, first_pos[0], first_pos[1]))
 
    if len(first_positions) < 4:
        return LineOfScrimmage.from_vertical_line(
            video_width / 2,
            video_height,
            confidence=0.0
        )
 
    x_values = np.array([p[1] for p in first_positions])
    y_values = np.array([p[2] for p in first_positions])
 
    if method == 'histogram':
        return _fit_using_histogram(x_values, y_values, video_width)
 
    elif method == 'clustering':
        return _fit_using_clustering(x_values, y_values, video_width)
 
    elif method == 'vertical':
        return _fit_using_vertical_separation(x_values, y_values, video_width)
 
    elif method == 'minimal_movement':
        return _fit_using_minimal_movement(trajectories, video_width, video_height, first_n_frames)
    else:
        raise ValueError(f"Unknown method: {method}")


def _fit_using_histogram(x_values: np.ndarray, y_values: np.ndarray,
                        video_width: float) -> LineOfScrimmage:
    """
    Fit LOS using histogram-based peak finding.
    """
    # Create histogram of x-values
    hist, bin_edges = np.histogram(x_values, bins=20)

    # Find the minimum between the two peaks
    # This gap is typically where the LOS lies
    peak1_idx = np.argmax(hist[:len(hist)//2])
    peak2_idx = np.argmax(hist[len(hist)//2:]) + len(hist)//2

    # Find minimum between peaks (the LOS region)
    gap_start = peak1_idx
    gap_end = peak2_idx
    min_density_idx = gap_start + np.argmin(hist[gap_start:gap_end]) if gap_end > gap_start else len(hist)//2

    # LOS is at the center of the gap
    los_x = (bin_edges[min_density_idx] + bin_edges[min_density_idx + 1]) / 2

    # Confidence based on how clear the separation is
    confidence = 1.0 - (hist[min_density_idx] / (hist[peak1_idx] + hist[peak2_idx] + 1e-6))

    return LineOfScrimmage.from_vertical_line(los_x, video_width, confidence=min(confidence, 1.0))


def _fit_using_clustering(x_values: np.ndarray, y_values: np.ndarray,
                          video_width: float) -> LineOfScrimmage:
    """
    Fit LOS using simple 2-cluster gap-finding.
    """
    # Sort by x-value (field length direction)
    sorted_indices = np.argsort(x_values)
    sorted_x = x_values[sorted_indices]

    # Find the largest gap between consecutive x-values
    x_diffs = np.diff(sorted_x)
    max_gap_idx = np.argmax(x_diffs)

    # LOS is in the middle of the largest gap
    los_x = (sorted_x[max_gap_idx] + sorted_x[max_gap_idx + 1]) / 2

    # Confidence based on gap size relative to field length
    gap_size = x_diffs[max_gap_idx]
    field_length = np.max(x_values) - np.min(x_values)
    confidence = min(gap_size / (field_length * 0.1), 1.0) if field_length > 0 else 0.5

    return LineOfScrimmage.from_vertical_line(los_x, video_width, confidence=confidence)


def _fit_using_vertical_separation(x_values: np.ndarray, y_values: np.ndarray,
                                   video_width: float) -> LineOfScrimmage:
    """
    Fit a vertical LOS assuming equal numbers of offense/defense players.

    Uses the median x-value as initial guess, then refines to balance counts.
    """
    # Initial estimate: median of x-values
    initial_los = np.median(x_values)

    # Count players on each side
    left = np.sum(x_values < initial_los)
    right = np.sum(x_values > initial_los)

    # Adjust LOS to balance the count
    if abs(int(left) - int(right)) > 2:
        # Try to find a better balance
        sorted_x = np.sort(x_values)
        n = len(sorted_x)

        # LOS should split roughly 11 vs 11 (or less if tracking issues)
        # Use percentile-based estimate
        los_x = np.percentile(sorted_x, 50)
    else:
        los_x = initial_los

    return LineOfScrimmage.from_vertical_line(los_x, video_width, confidence=0.8)

def _fit_using_minimal_movement(trajectories: List[ParsedTrajectory],
                                video_width: float,
                                video_height: float,
                                first_n_frames: int = 15) -> 'LineOfScrimmage':
    """
    Fit LOS by identifying the cluster of players with minimal X movement
    over the first N frames
    """

    player_stats = []

    for traj in trajectories:
        xs = []
        ys = []
        # FIX: detections is a list of Detection objects, not a dict
        for det in traj.detections[:first_n_frames]:
            xs.append(det.center[0])   # use the center property
            ys.append(det.center[1])

        if len(xs) < 3:
            continue

        x_var = float(np.var(xs))
        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        player_stats.append((traj.track_id, mean_x, mean_y, x_var))
 
    if len(player_stats) < 6:
        # Fallback: not enough data, use simple clustering on first positions
        x_vals = np.array([p[1] for p in player_stats]) if player_stats else np.array([video_width / 2])
        y_vals = np.array([p[2] for p in player_stats]) if player_stats else np.array([video_height / 2])
        return _fit_using_clustering(x_vals, y_vals, video_width)
 
    player_stats.sort(key=lambda p: p[3])
 
    # We expect at minimum 5 OL + 4-5 DL = 9-10 players in the cluster.
    # Take up to 12 to also absorb a blocking TE or QB under center.
    # But cap at len(player_stats) - 2 so we always leave some non-linemen out.
    n_total = len(player_stats)
    k = min(12, max(6, n_total - 4))
    line_cluster = player_stats[:k]
 
    cluster_xs = np.array([p[1] for p in line_cluster])
    cluster_ys = np.array([p[2] for p in line_cluster])
 
    if len(cluster_ys) >= 2 and np.std(cluster_ys) > 1.0:
        coeffs = np.polyfit(cluster_ys, cluster_xs, 1)   # [slope, intercept]
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
    else:
        # Degenerate case — all players at same Y, just use mean X
        slope = 0.0
        intercept = float(np.mean(cluster_xs))
 
    # ------------------------------------------------------------------
    all_xs = np.array([p[1] for p in player_stats])
    all_ys = np.array([p[2] for p in player_stats])
 
    # Project each player onto the line and get signed X residuals
    projected_x = slope * all_ys + intercept
    residuals = all_xs - projected_x  # positive = offense side
 
    # Sort residuals; we want 11 players with positive residual.
    # The intercept shift needed = -(11th largest residual), i.e. we slide
    # the line so that exactly 11 residuals are positive.
    n_players = len(residuals)
    target_offense = 11  # exactly one team's worth
 
    if n_players >= target_offense:
        sorted_residuals = np.sort(residuals)[::-1]  # descending
        # The threshold: 11th largest residual should be just > 0
        # So shift intercept by that residual to put exactly 11 on offense side
        threshold = sorted_residuals[target_offense - 1]
        intercept_adjusted = intercept + threshold
    else:
        # Fewer than 11 players tracked — just use the cluster centroid
        intercept_adjusted = intercept
 
    # Compare variance of line-cluster vs full set
    cluster_x_var = float(np.var(cluster_xs))
    all_x_var = float(np.var(all_xs))
    # If the cluster is much less spread than everyone, confidence is high
    confidence = min(1.0, 1.0 - (cluster_x_var / (all_x_var + 1e-6)))
    confidence = max(0.0, confidence)
 
    return LineOfScrimmage.from_sloped_line(
        slope=slope,
        intercept=intercept_adjusted,
        video_height=video_height,
        confidence=confidence
    )


class LineOfScrimmageClassifier:
    """
    Classifier for distinguishing offensive and defensive players.
    """

    def __init__(self, method: str = 'clustering'):
        """
        Initialize the classifier.

        Args:
            method: Method for LOS estimation ('clustering', 'histogram', 'vertical')
        """
        self.method = method
        self.los: Optional[LineOfScrimmage] = None
        self.classifications: dict = {}

    def fit(self, trajectories: List[ParsedTrajectory],
            video_width: float = 1920,
            video_height: float = 1080,
            first_n_frames: int = 15) -> 'LineOfScrimmageClassifier':
        """
        Fit the classifier to trajectories from a single video.
        """
        self.los = fit_line_of_scrimmage(trajectories, method=self.method, video_width=video_width, video_height=video_height, first_n_frames=first_n_frames)
        return self

    def classify(self, trajectory: ParsedTrajectory) -> str:
        """
        Classify a single trajectory as 'offense' or 'defense'.
        """
        if self.los is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        first_pos = trajectory.get_first_position()
        if first_pos is None:
            return 'unknown'

        x, y = first_pos
        team = self.los.classify_position(x, y)

        self.classifications[trajectory.track_id] = team
        return team

    def classify_all(self, trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """
        Classify all trajectories and update their team labels.
        """
        for traj in trajectories:
            traj.team = self.classify(traj)
        return trajectories

    def get_offense_trajectories(self, trajectories: List[ParsedTrajectory]
                                 ) -> List[ParsedTrajectory]:
        """Get only offensive trajectories."""
        return [t for t in trajectories if t.team == 'offense']

    def get_defense_trajectories(self, trajectories: List[ParsedTrajectory]
                                 ) -> List[ParsedTrajectory]:
        """Get only defensive trajectories."""
        return [t for t in trajectories if t.team == 'defense']


# HELPERRRR

def classify_offense_defense(trajectories: List[ParsedTrajectory],
                             method: str = 'minimal_movement',
                             video_width: float = 1920,
                             video_height: float = 1080,
                             first_n_frames: int = 15
                             ) -> Tuple[List[ParsedTrajectory], LineOfScrimmage]:
    """
    Convenience function to classify trajectories as offense/defense.
    """
    classifier = LineOfScrimmageClassifier(method=method)
    classifier.fit(trajectories, video_width, video_height, first_n_frames)
    classifier.classify_all(trajectories)
    return trajectories, classifier.los
