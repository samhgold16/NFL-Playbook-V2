"""NFL Route Tracker - Phase 2: Skill Position Filtering
======================================================
Filters offensive trajectories to identify skill position players (WR, TE, RB).
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .data_loader import ParsedTrajectory
from .offense_defense_classifier import LineOfScrimmage


class PlayerClassification(Enum):
    """
    Player position classifications.
    """
    WIDE_RECEIVER = 'WR'
    TIGHT_END = 'TE'
    RUNNING_BACK = 'RB'
    OFFENSIVE_LINE = 'OL'
    QUARTERBACK = 'QB'
    FULLBACK = 'FB'
    DEFENSIVE_LINE = 'DL'
    LINEBACKER = 'LB'
    DEFENSIVE_BACK = 'DB'
    UNKNOWN = 'UNKNOWN'

    @property
    def is_skill_position(self) -> bool:
        """Returns True if this is a skill position player."""
        return self in {PlayerClassification.WIDE_RECEIVER,
                        PlayerClassification.TIGHT_END,
                        PlayerClassification.RUNNING_BACK}


@dataclass
class PositionConfidence:
    """
    Confidence scores for position classification.
    """
    position: PlayerClassification
    confidence: float  # 0.0 to 1.0
    reasons: List[str] = None  # Human-readable reasons for classification

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class SkillPositionFilter:
    """
    Filters offensive trajectories to identify skill position players.
    """

    # Minimum X-distance beyond LOS to classify as skill position
    # Positive X = moving toward opponent's endzone (upfield)
    DEFAULT_LOS_BUFFER = 100.0

    DEFAULT_SIDELINE_PCT = 0.20

    # Minimum trajectory length to consider
    DEFAULT_MIN_DETECTIONS = 30


    def __init__(self, los: LineOfScrimmage, los_buffer: float = DEFAULT_LOS_BUFFER, sideline_pct: float = DEFAULT_SIDELINE_PCT, min_detections: int = DEFAULT_MIN_DETECTIONS):
        """
        Initialize the filter.
        """
        self.los = los
        self.los_buffer = los_buffer
        self.sideline_pct = sideline_pct
        self.min_detections = min_detections

    def is_skill_position(self, trajectory: ParsedTrajectory) -> Tuple[bool, str]:
        """
        Determine whether a single trajectory belongs to a skill position player.
        """
        if trajectory.num_detections < self.min_detections:
            return False, "Trajectory too short for classification"

        first_pos = trajectory.get_first_position()
        last_pos = trajectory.get_last_position()
        if first_pos is None or last_pos is None:
            return False, "Missing position data"

        first_x, first_y = first_pos
        last_x, last_y = last_pos

        net_x_displacement = last_x - first_x  # positive = moved upfield (right)

        if net_x_displacement <= -50:
            return False, (
                f"Net X displacement is negative ({net_x_displacement:.0f}px) — "
                f"player moved backward, cannot be skill position"
            )


        # ------------------------------------------------------------------
        if net_x_displacement > self.los_buffer:
            return True, (
                f"Positive net X displacement of {net_x_displacement:.0f}px "
                f"(threshold={self.los_buffer:.0f}px) → route runner"
            )

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        los_x = self.los.get_x_at_y(first_y)
        video_height = trajectory.video_height

        if first_x <= los_x:
            near_bottom = first_y < self.sideline_pct * video_height
            near_top    = first_y > (1.0 - self.sideline_pct) * video_height
            if near_bottom or near_top:
                side = "bottom" if near_bottom else "top"
                norm_y = first_y / video_height if video_height > 0 else 0.5
                return True, (
                    f"Behind LOS + near {side} sideline "
                    f"(y_norm={norm_y:.2f}) → backfield skill player"
                )

        return False, (
            f"Net X displacement {net_x_displacement:.0f}px "
            f"(not positive enough) → likely OL/QB"
        )

    def classify_position(self, trajectory: ParsedTrajectory) -> PositionConfidence:
        """
        Classify into WR / TE / RB / OL / QB using X displacement and Y alignment.
        """
        skill, reason = self.is_skill_position(trajectory)

        if not skill:
            total_dist = trajectory.total_distance
            if total_dist < 300:
                return PositionConfidence(PlayerClassification.OFFENSIVE_LINE, 0.7,
                                        [reason, "Low total movement → OL"])
            else:
                return PositionConfidence(PlayerClassification.QUARTERBACK, 0.55,
                                        [reason, f"Moderate movement ({total_dist:.0f}px) → QB/FB"])

        first_pos = trajectory.get_first_position()
        last_pos  = trajectory.get_last_position()
        x, y      = first_pos
        last_x, _ = last_pos
        los_x     = self.los.get_x_at_y(y)
        video_height = trajectory.video_height
        norm_y    = y / video_height if video_height > 0 else 0.5

        behind_los   = x <= los_x
        near_sideline = norm_y < self.sideline_pct or norm_y > (1.0 - self.sideline_pct)

        if behind_los:
            # Started behind LOS + near sideline → RB in flat/swing
            return PositionConfidence(
                PlayerClassification.RUNNING_BACK, 0.65,
                [reason, "Behind LOS + near sideline → RB"]
            )

        if near_sideline:
            # Upfield + near sideline → WR
            return PositionConfidence(
                PlayerClassification.WIDE_RECEIVER, 0.75,
                [reason, f"Upfield + sideline alignment (y_norm={norm_y:.2f}) → WR"]
            )

        # Upfield + mid-frame → TE or slot WR
        mid_dist = min(norm_y, 1.0 - norm_y)  # 0 = sideline, 0.5 = center
        if mid_dist < 0.30:
            return PositionConfidence(
                PlayerClassification.WIDE_RECEIVER, 0.60,
                [reason, f"Upfield + near-slot (y_norm={norm_y:.2f}) → slot WR"]
            )
        else:
            return PositionConfidence(
                PlayerClassification.TIGHT_END, 0.60,
                [reason, f"Upfield + inline/center (y_norm={norm_y:.2f}) → TE"]
            )


    def filter_skill_positions(self, trajectories: List[ParsedTrajectory]) -> Tuple[List[ParsedTrajectory], List[ParsedTrajectory]]:
        """
        Filter trajectories into skill position and non-skill position.
        """
        skill_positions = []
        non_skill = []

        for traj in trajectories:
            if traj.team != 'offense':
                continue
 
            pos_conf = self.classify_position(traj)
            traj.position = pos_conf.position.value
            traj.position_confidence = pos_conf.confidence
            traj.position_reasons = pos_conf.reasons
            traj.is_skill_position = pos_conf.position.is_skill_position
 
            if traj.is_skill_position:
                skill_positions.append(traj)
            else:
                non_skill.append(traj)
 
        return skill_positions, non_skill

    def get_wrs(self, trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """Get only Wide Receiver trajectories."""
        return [t for t in trajectories
                if t.position == PlayerClassification.WIDE_RECEIVER.value]

    def get_tes(self, trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """Get only Tight End trajectories."""
        return [t for t in trajectories
                if t.position == PlayerClassification.TIGHT_END.value]

    def get_rbs(self, trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """Get only Running Back trajectories."""
        return [t for t in trajectories
                if t.position == PlayerClassification.RUNNING_BACK.value]


def filter_skill_position_players(trajectories: List[ParsedTrajectory], los: LineOfScrimmage,
                                  los_buffer: float = 200.0, sideline_pct: float = .20) -> Tuple[List[ParsedTrajectory], List[ParsedTrajectory]]:
    """
    Convenience function to filter skill position players.
    """
    f = SkillPositionFilter(los, los_buffer=los_buffer, sideline_pct=sideline_pct)
    return f.filter_skill_positions(trajectories)
