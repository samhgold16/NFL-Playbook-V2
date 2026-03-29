"""
NFL Route Tracker - Temporal Detection Aggregator
===============================================

This module aggregates detections across multiple frames to reduce noise and improve tracking stability.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from nfl_route_tracker.detection.player_detector import DetectionResult

@dataclass
class TemporalAggregator:
    """
    Aggregates detections across multiple frames for more stable tracking.
    """

    def __init__(self, window_size: int = 3, stride: int = 1, method: str = 'mean',
                 confidence_weight: float = 0.6, min_detection_count: int = 2, position_threshold: float = 30.0):
        """
        Initialize the temporal aggregator.
        """
        self.window_size = window_size # num frames to consider
        self.stride = stride # step size between frames
        self.method = method # how to aggregate (mean, max, weighted)
        self.confidence_weight = confidence_weight # weight for confidence in weighted method
        self.min_detection_count = min_detection_count # minimum frames a detection must appear in to be kept
        self.position_threshold = position_threshold # max distance in pixels to consider detections the same

        # Frame buffer: frame_id -> list of detections
        self._buffer: Dict[int, List[DetectionResult]] = {}

        # Next frame to process
        self._next_frame_id: int = 0

        # Track how many times each detection position appeared
        self._detection_counts: Dict[int, int] = defaultdict(int)

    def add_frame(self, frame_id: int, detections: List[DetectionResult]) -> None:
        """
        Add a new frame's detections to the buffer.
        """
        self._buffer[frame_id] = detections.copy()
        self._next_frame_id = frame_id + 1

        # Remove old frames from buffer (keep only window_size frames)
        min_frame = frame_id - self.window_size + 1
        self._buffer = {fid: dets for fid, dets in self._buffer.items() if fid >= min_frame}

    def get_aggregated(self, frame_id: int) -> List[DetectionResult]:
        """
        Get aggregated detections for a specific frame.
        """
        # Collect frames in window
        window_frames = []
        for fid in range(max(0, frame_id - self.window_size + 1), frame_id + 1):
            if fid in self._buffer:
                window_frames.append(fid)

        if len(window_frames) == 0:
            return []

        # Get all detections from window
        all_detections = []
        for fid in window_frames:
            all_detections.append((fid, self._buffer[fid]))

        # Aggregate detections, aggregating functions implemented below
        if self.method == 'mean':
            return self._aggregate_mean(all_detections, frame_id)
        elif self.method == 'max':
            return self._aggregate_max(all_detections, frame_id)
        elif self.method == 'weighted':
            return self._aggregate_weighted(all_detections, frame_id)
        else:
            # Default to current frame
            return self._buffer.get(frame_id, []).copy()
        
    # helper functions below to cluster detections and compute aggregates
    # all helper functions implemented below

    def _aggregate_mean(self, all_detections: List[Tuple[int, List[DetectionResult]]], target_frame: int) -> List[DetectionResult]:
        """Aggregate using mean position."""
        # Cluster detections by position
        clusters = self._cluster_detections(all_detections)

        aggregated = []
        for cluster_dets in clusters.values():
            if len(cluster_dets) >= self.min_detection_count:
                agg = self._mean_detection(cluster_dets, target_frame)
                if agg is not None:
                    aggregated.append(agg)

        return aggregated

    def _aggregate_max(self,  all_detections: List[Tuple[int, List[DetectionResult]]], target_frame: int) -> List[DetectionResult]:
        """Aggregate using max confidence."""
        clusters = self._cluster_detections(all_detections)

        aggregated = []
        for cluster_dets in clusters.values():
            if len(cluster_dets) >= self.min_detection_count:
                # Take detection with highest confidence
                best = max(cluster_dets, key=lambda d: d.confidence)
                aggregated.append(best)

        return aggregated

    def _aggregate_weighted(self,
                          all_detections: List[Tuple[int, List[DetectionResult]]],
                          target_frame: int) -> List[DetectionResult]:
        """Aggregate using weighted mean of position and confidence."""
        clusters = self._cluster_detections(all_detections)

        aggregated = []
        for cluster_dets in clusters.values():
            if len(cluster_dets) >= self.min_detection_count:
                agg = self._weighted_detection(cluster_dets, target_frame)
                if agg is not None:
                    aggregated.append(agg)

        return aggregated
    
    def _cluster_detections(self, all_detections: List[Tuple[int, List[DetectionResult]]]) -> Dict[int, List[DetectionResult]]:
        """
        Cluster detections across frames by position.
        """
        clusters = {}  
        next_cluster_id = 0

        for frame_id, detections in all_detections:
            for det in detections:
                # Try to match existing cluster
                matched = False
                cx, cy = det.center

                for cid, cluster_dets in clusters.items():
                    # Check distance to cluster centroid
                    if cluster_dets:
                        cluster_cx = np.mean([d.center[0] for d in cluster_dets])
                        cluster_cy = np.mean([d.center[1] for d in cluster_dets])

                        dist = np.sqrt((cx - cluster_cx)**2 + (cy - cluster_cy)**2)

                        if dist < self.position_threshold:
                            clusters[cid].append(det)
                            matched = True
                            break

                if not matched:
                    clusters[next_cluster_id] = [det]
                    next_cluster_id += 1

        return clusters
    
    def _mean_detection(self, detections: List[DetectionResult], target_frame: int) -> Optional[DetectionResult]:
        """Compute mean detection from a cluster."""
        if not detections:
            return None

        # Average position
        mean_x = np.mean([d.x for d in detections])
        mean_y = np.mean([d.y for d in detections])
        mean_w = np.mean([d.width for d in detections])
        mean_h = np.mean([d.height for d in detections])

        # Average confidence
        mean_conf = np.mean([d.confidence for d in detections])

        return DetectionResult(x = mean_x, y = mean_y, width = mean_w,
                               height = mean_h, confidence = mean_conf, class_id = detections[0].class_id,
                               class_name = detections[0].class_name)

    def _weighted_detection(self,
                          detections: List[DetectionResult],
                          target_frame: int) -> Optional[DetectionResult]:
        """
        Compute weighted mean detection.
        Weights position by confidence.
        """
        if not detections:
            return None

        # Compute weights (confidence-based)
        weights = np.array([d.confidence for d in detections])
        weights = weights / weights.sum()  # Normalize

        # Weighted position
        weighted_x = sum(d.x * w for d, w in zip(detections, weights))
        weighted_y = sum(d.y * w for d, w in zip(detections, weights))
        weighted_w = sum(d.width * w for d, w in zip(detections, weights))
        weighted_h = sum(d.height * w for d, w in zip(detections, weights))

        # Max confidence (for confidence score)
        max_conf = max(d.confidence for d in detections)

        return DetectionResult(x = weighted_x, y = weighted_y, width = weighted_w,
                               height = weighted_h, confidence = max_conf, class_id = detections[0].class_id,
                               class_name = detections[0].class_name)
    
    def clear(self) -> None:
        """Clear the buffer and reset state."""
        self._buffer.clear()
        self._detection_counts.clear()
        self._next_frame_id = 0

    def reset(self) -> None:
        """Alias for clear() for API consistency."""
        self.clear()

# Convenience function for simple temporal smoothing
# check if used anywhere,  need this????
def smooth_detections(detections: List[DetectionResult],
                      prev_detections: List[DetectionResult],
                      alpha: float = 0.7) -> List[DetectionResult]:
    """
    Simple one-frame smoothing (exponential moving average).
    """
    if not prev_detections:
        return detections

    smoothed = []

    for det in detections:
        cx, cy = det.center

        # Find closest previous detection
        best_dist = float('inf')
        best_prev = None

        for prev in prev_detections:
            pcx, pcy = prev.center
            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)

            # Only consider if within reasonable distance (50 pixels)
            if dist < 50 and dist < best_dist:
                best_dist = dist
                best_prev = prev

        if best_prev is not None:
            # Smooth the position
            pcx, pcy = best_prev.center
            new_cx = alpha * cx + (1 - alpha) * pcx
            new_cy = alpha * cy + (1 - alpha) * pcy

            # Adjust bbox to keep same size, new center
            new_x = new_cx - det.width / 2
            new_y = new_cy - det.height / 2

            smoothed.append(DetectionResult(x = new_x, y = new_y, width = det.width,
                                            height = det.height, confidence = det.confidence, class_id = det.class_id,
                                            class_name = det.class_name))
        else:
            smoothed.append(det)

    return smoothed