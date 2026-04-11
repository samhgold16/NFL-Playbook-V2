"""
NFL Route Tracker - Trajectory Post-Processing Module
====================================================

This module handles post-processing of trajectories to merge fragmented tracks.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import math

from nfl_route_tracker.tracking.trajectory import Trajectory, TrajectoryStore, Detection

@dataclass
class MergeCandidate:
    """Represents a potential merge between two trajectory fragments."""
    traj_a_id: int
    traj_b_id: int
    spatial_distance: float  # Distance between end of A and start of B
    temporal_gap: int  # Frames between end of A and start of B
    direction_score: float  # How consistent the direction is (0-1)
    size_similarity: float  # How similar the bbox sizes are (0-1)
    confidence_score: float  # Combined score for merge decision

class TrajectoryMerger:
    """
    Merges fragmented trajectories based on spatial-temporal continuity.
    """

    def __init__(self, spatial_threshold: float = 150.0,  # pixels - max distance between fragments
                temporal_threshold: int = 45,  # frames - max gap to consider merging
                direction_weight: float = 0.3,
                size_weight: float = 0.3,
                confidence_threshold: float = 0.5,  # Minimum score to perform merge
                min_merge_length: int = 10,
                density_radius: float = 200.0,      # px — neighbourhood radius for crowd check
                density_threshold: int = 4,         # endpoints in radius → region is crowded
                max_merges: int = 2):
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.direction_weight = direction_weight
        self.size_weight = size_weight
        self.confidence_threshold = confidence_threshold
        self.min_merge_length = min_merge_length
        self.density_radius = density_radius
        self.density_threshold = density_threshold
        self.max_merges = max_merges

    def merge_trajectories(self, store: TrajectoryStore) -> TrajectoryStore:
        """
        Merge fragmented trajectories in a TrajectoryStore.
        Returns a new TrajectoryStore with merged trajectories.
        """
        trajectories = store.get_all_trajectories()

        if len(trajectories) < 2:
            return store

        # Filter out very short trajectories
        valid_trajectories = [t for t in trajectories if len(t) >= self.min_merge_length]

        all_endpoints = self._collect_endpoints(valid_trajectories)

        # Find all merge candidates
        merge_counts: Dict[int, int] = {t.track_id: 0 for t in valid_trajectories}
        candidates = self._find_merge_candidates(valid_trajectories, all_endpoints, merge_counts)

        if not candidates:
            print("No trajectory merges found")
            return store

        # Build merge groups using union-find
        merge_groups = self._build_merge_groups(candidates, len(valid_trajectories))

        # Create new trajectory store with merged trajectories
        merged_store = TrajectoryStore()
        used_ids: Set[int] = set() # old_id -> new_id

        # Track which original IDs have been merged
        used_ids = set()

        for traj in valid_trajectories:
            if traj.track_id in used_ids:
                continue

            # Check if this trajectory is part of a merge group
            if traj.track_id in merge_groups:
                group = merge_groups[traj.track_id]
                merged_traj = self._merge_trajectory_group([t for t in valid_trajectories if t.track_id in group])
                new_id = merged_traj.track_id
                merged_store._trajectories[new_id] = merged_traj
                merged_store._total_detections += len(merged_traj)
                used_ids.update(group)
            else:
                # Keep trajectory as-is
                merged_store._trajectories[traj.track_id] = traj
                merged_store._total_detections += len(traj)
                used_ids.add(traj.track_id)

        # Add back very short trajectories
        short_trajectories = [t for t in trajectories if len(t) < self.min_merge_length]
        for traj in short_trajectories:
            # Generate new unique ID
            new_id = max(list(merged_store._trajectories.keys()) + [0]) + 1
            traj.track_id = new_id
            merged_store._trajectories[new_id] = traj
            merged_store._total_detections += len(traj)

        original_count = len(trajectories)
        merged_count = merged_store.num_trajectories
        print(f"\nTrajectory Post-Processing:")
        print(f"  Original trajectories: {original_count}")
        print(f"  After merging: {merged_count}")
        print(f"  Merges performed: {original_count - merged_count}")

        return merged_store
    
    def _collect_endpoints(
        self, trajectories: List[Trajectory]
    ) -> List[Tuple[float, float]]:
        """
        Return every trajectory start- and end-position as a flat list.
 
        This list is used by the crowd density filter to cheaply estimate how
        many players are beginning or finishing their routes near any given point.
        """
        pts: List[Tuple[float, float]] = []
        for t in trajectories:
            pts.append(self._get_start_position(t))
            pts.append(self._get_end_position(t))
        return pts
 
    def _count_nearby_endpoints(
        self,
        point: Tuple[float, float],
        all_endpoints: List[Tuple[float, float]],
        radius: float,
    ) -> int:
        """Count how many endpoints fall within `radius` pixels of `point`."""
        px, py = point
        count = 0
        for ex, ey in all_endpoints:
            if math.sqrt((px - ex) ** 2 + (py - ey) ** 2) <= radius:
                count += 1
        return count
    
    def _is_crowded(
        self,
        join_point: Tuple[float, float],
        all_endpoints: List[Tuple[float, float]],
    ) -> bool:
        """
        Return True when the neighbourhood around `join_point` contains too
        many endpoints to safely infer that any two specific trajectories belong
        to the same player.
 
        The join-point is the midpoint between the end of fragment A and the
        start of fragment B — the location where the hypothetical merge would
        occur.
        """
        nearby = self._count_nearby_endpoints(
            join_point, all_endpoints, self.density_radius
        )
        return nearby > self.density_threshold
    
    def _find_merge_candidates(
        self,
        trajectories: List[Trajectory],
        all_endpoints: List[Tuple[float, float]],
        merge_counts: Dict[int, int],
    ) -> List[MergeCandidate]:
        """
        Find all pairs of trajectories that could be merged.
 
        Two pre-filters run before the expensive scoring:
          1. Crowd density — skip if the join region is congested.
          2. Chain limit   — skip if either trajectory has been merged too often.
        """
        candidates: List[MergeCandidate] = []
        sorted_trajs = sorted(trajectories, key=lambda t: t.get_frame_range()[0])
 
        for i, traj_a in enumerate(sorted_trajs):
            end_frame_a = traj_a.get_frame_range()[1]
            end_pos_a = self._get_end_position(traj_a)
 
            for traj_b in sorted_trajs[i + 1:]:
                start_frame_b = traj_b.get_frame_range()[0]
 
                if start_frame_b <= end_frame_a:
                    continue
 
                temporal_gap = start_frame_b - end_frame_a
 
                if temporal_gap > self.temporal_threshold:
                    break  # sorted — remaining gaps are larger
 
                start_pos_b = self._get_start_position(traj_b)
                spatial_distance = self._euclidean_distance(end_pos_a, start_pos_b)
 
                if spatial_distance > self.spatial_threshold:
                    continue
 
                # ---- Filter 1: crowd density --------------------------------
                # The join-point is the midpoint between the two fragment edges.
                join_point = (
                    (end_pos_a[0] + start_pos_b[0]) / 2.0,
                    (end_pos_a[1] + start_pos_b[1]) / 2.0,
                )
                if self._is_crowded(join_point, all_endpoints):
                    continue  # too many players nearby — skip this pair
 
                # ---- Filter 2: merge-chain limit ----------------------------
                # Reject if either participant has already hit the merge cap.
                if (
                    merge_counts.get(traj_a.track_id, 0) >= self.max_merges
                    or merge_counts.get(traj_b.track_id, 0) >= self.max_merges
                ):
                    continue
 
                # ---- Confidence scoring (unchanged from v1) -----------------
                direction_score = self._calculate_direction_score(traj_a, traj_b)
                size_similarity = self._calculate_size_similarity(traj_a, traj_b)
 
                spatial_score = 1.0 - (spatial_distance / self.spatial_threshold)
                temporal_score = 1.0 - (temporal_gap / self.temporal_threshold)
                confidence = (
                    0.4 * spatial_score
                    + 0.3 * temporal_score
                    + self.direction_weight * direction_score
                    + self.size_weight * size_similarity
                )
 
                if confidence >= self.confidence_threshold:
                    candidates.append(
                        MergeCandidate(
                            traj_a_id=traj_a.track_id,
                            traj_b_id=traj_b.track_id,
                            spatial_distance=spatial_distance,
                            temporal_gap=temporal_gap,
                            direction_score=direction_score,
                            size_similarity=size_similarity,
                            confidence_score=confidence,
                        )
                    )
                    # Update merge counts so later iterations respect the limit.
                    merge_counts[traj_a.track_id] = (
                        merge_counts.get(traj_a.track_id, 0) + 1
                    )
                    merge_counts[traj_b.track_id] = (
                        merge_counts.get(traj_b.track_id, 0) + 1
                    )
 
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        return candidates
 
    # ------------------------------------------------------------------
    # Union-find grouping (unchanged)
    # ------------------------------------------------------------------
 
    def _build_merge_groups(
        self, candidates: List[MergeCandidate], num_trajs: int
    ) -> Dict[int, Set[int]]:
        """
        Build merge groups using union-find.
        Returns a dict mapping trajectory ID → set of all IDs it merges with.
        """
        id_set: Set[int] = set()
        for c in candidates:
            id_set.add(c.traj_a_id)
            id_set.add(c.traj_b_id)
        track_ids = sorted(list(id_set))
        id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
        parent = list(range(len(track_ids)))
 
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
 
        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
 
        for c in candidates:
            union(id_to_idx[c.traj_a_id], id_to_idx[c.traj_b_id])
 
        groups: Dict[int, Set[int]] = {}
        for tid in track_ids:
            root = track_ids[find(id_to_idx[tid])]
            groups.setdefault(root, set()).add(tid)
 
        return groups
 
    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------
 
    def _merge_trajectory_group(self, trajectories: List[Trajectory]) -> Trajectory:
        """
        Merge a group of temporally non-overlapping trajectories into one.
        Uses the smallest track_id as the canonical ID.
        """
        if len(trajectories) == 1:
            return trajectories[0]
 
        sorted_trajs = sorted(trajectories, key=lambda t: t.get_frame_range()[0])
        new_id = min(t.track_id for t in sorted_trajs)
        merged = Trajectory(track_id=new_id)
 
        for traj in sorted_trajs:
            for det in traj.detections:
                merged.add_detection(det)
 
        return merged
 
    # ------------------------------------------------------------------
    # Geometry helpers (unchanged)
    # ------------------------------------------------------------------
 
    def _get_start_position(self, traj: Trajectory) -> Tuple[float, float]:
        if not traj.detections:
            return (0.0, 0.0)
        return traj.detections[0].center
 
    def _get_end_position(self, traj: Trajectory) -> Tuple[float, float]:
        if not traj.detections:
            return (0.0, 0.0)
        return traj.detections[-1].center
 
    def _euclidean_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
 
    def _calculate_direction_score(
        self, traj_a: Trajectory, traj_b: Trajectory
    ) -> float:
        frames_a, xs_a, ys_a = traj_a.get_path()
        if len(frames_a) < 2:
            return 0.5
        dx_a, dy_a = xs_a[-1] - xs_a[-2], ys_a[-1] - ys_a[-2]
 
        frames_b, xs_b, ys_b = traj_b.get_path()
        if len(frames_b) < 2:
            return 0.5
        dx_b, dy_b = xs_b[1] - xs_b[0], ys_b[1] - ys_b[0]
 
        mag_a = math.sqrt(dx_a ** 2 + dy_a ** 2)
        mag_b = math.sqrt(dx_b ** 2 + dy_b ** 2)
        if mag_a < 1e-6 or mag_b < 1e-6:
            return 0.5
 
        cos_sim = (dx_a * dx_b + dy_a * dy_b) / (mag_a * mag_b)
        return (cos_sim + 1.0) / 2.0
 
    def _calculate_size_similarity(
        self, traj_a: Trajectory, traj_b: Trajectory
    ) -> float:
        end_areas_a = (
            [traj_a.detections[0].area]
            if len(traj_a.detections) < 2
            else [d.area for d in traj_a.detections[-3:]]
        )
        start_areas_b = (
            [traj_b.detections[0].area]
            if len(traj_b.detections) < 2
            else [d.area for d in traj_b.detections[:3]]
        )
        avg_a = sum(end_areas_a) / len(end_areas_a)
        avg_b = sum(start_areas_b) / len(start_areas_b)
        if avg_a < 1e-6:
            return 0.5
        return min(avg_a, avg_b) / max(avg_a, avg_b)
 
 
# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------
 
def merge_trajectory_store(
    store: TrajectoryStore,
    spatial_threshold: float = 150.0,
    temporal_threshold: int = 45,
    confidence_threshold: float = 0.5,
    density_radius: float = 200.0,
    density_threshold: int = 4,
    max_merges: int = 2,
) -> TrajectoryStore:
    """Convenience function to merge trajectories in a TrajectoryStore."""
    merger = TrajectoryMerger(
        spatial_threshold=spatial_threshold,
        temporal_threshold=temporal_threshold,
        confidence_threshold=confidence_threshold,
        density_radius=density_radius,
        density_threshold=density_threshold,
        max_merges=max_merges,
    )
    return merger.merge_trajectories(store)
 