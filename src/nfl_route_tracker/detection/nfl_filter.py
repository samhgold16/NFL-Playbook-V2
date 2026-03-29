"""
NFL Route Tracker - NFL-Specific Detection Filter
=================================================

This module provides NFL-specific post-processing for YOLO detections.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2

from nfl_route_tracker.detection.player_detector import DetectionResult

# same as NFLDetectionFilterConfig in config.py file
# but with methods to apply the filter to detections, and to validate detections based on NFL-specific heuristics (size, aspect ratio, position on field)
@dataclass
class NFLDetectionFilter:
    """
    Filters and validates YOLO detections specifically for NFL All-22 footage.
    """

    # Bounding box area constraints (width * height in pixels), depends on camera zoom
    min_area: int = 400
    max_area: int = 20000

    # Aspect ratio constraints (width / height)
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 1.75

    # Confidence thresholds
    min_confidence: float = 0.15  # Below this, ignore detection
    low_confidence: float = 0.05  # Below this, use ByteTrack association

    # Field zone thresholds (y-position in frame)
    # All-22 camera angle: top = far, bottom = near
    near_y_threshold: int = 734   # Players below this are "near"
    far_y_threshold: int = 250    # Players above this are "far"

    # Near players: larger, wider (closer to camera)
    near_area_range: Tuple[int, int] = (600, 20000)
    near_aspect_range: Tuple[float, float] = (0.2, 1.75)

    # Far players: smaller, taller (further from camera)
    far_area_range: Tuple[int, int] = (200, 8000)
    far_aspect_range: Tuple[float, float] = (0.2, 1.75)

    # Mid-range players
    mid_area_range: Tuple[int, int] = (400, 12000)
    mid_aspect_range: Tuple[float, float] = (0.25, 1.75)

    # Vertical position constraints (y-range in frame)
    # Players should be within these bounds
    min_y_position: int = 25     # Too high = likely crowd/noise
    max_y_position: int = 959     # Too low = likely camera artifact

    merge_iou_threshold: float = 0.5

    def filter_detections(self, detections: List[DetectionResult], frame_height: int = 984) -> List[DetectionResult]:
        """
        Apply all NFL-specific filters to detections.
        """
        filtered = []

        # apply fltering for each detection
        for det in detections:
            # confidence filter
            if det.confidence < self.min_confidence:
                continue

            # area filter
            area = det.area
            if area < self.min_area or area > self.max_area:
                continue

            # aspect ratio filter
            aspect = det.width / det.height if det.height > 0 else 0
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue

            # vertical position filter
            if det.y < self.min_y_position or det.y > self.max_y_position:
                continue

            # field zone-specific filtering, _check_field_zone() coded later
            if not self._check_field_zone(det, frame_height):
                continue

            filtered.append(det)

        return filtered

    def _check_field_zone(self, det: DetectionResult, frame_height: int) -> bool:
        """
        Check if detection is valid for its field zone.
        Players in different parts of the frame have different expected sizes
        """
        area = det.area
        aspect = det.width / det.height if det.height > 0 else 0
        center_y = det.y + det.height / 2

        # Determine zone based on y-position
        if center_y >= self.near_y_threshold:
            # Near zone (bottom of frame)
            if not (self.near_area_range[0] <= area <= self.near_area_range[1]):
                return False
            if not (self.near_aspect_range[0] <= aspect <= self.near_aspect_range[1]):
                return False
        elif center_y <= self.far_y_threshold:
            # Far zone (top of frame)
            if not (self.far_area_range[0] <= area <= self.far_area_range[1]):
                return False
            if not (self.far_aspect_range[0] <= aspect <= self.far_aspect_range[1]):
                return False
        else:
            # Middle zone
            if not (self.mid_area_range[0] <= area <= self.mid_area_range[1]):
                return False
            if not (self.mid_aspect_range[0] <= aspect <= self.mid_aspect_range[1]):
                return False

        return True

    def merge_overlapping_detections(self, detections: List[DetectionResult],  iou_threshold: float = 0.3) -> List[DetectionResult]:
        """
        Handle overlapping detections by merging or suppressing them based on IOU and confidence.
        """
        if iou_threshold is None:
            iou_threshold = self.merge_iou_threshold

        if len(detections) <= 1:
            return detections

        # Calculate pairwise IOU
        keep_detections = []
        suppress_indices = set()

        det_boxes = np.array([[d.x, d.y, d.x + d.width, d.y + d.height]
                            for d in detections])

        for i in range(len(detections)):
            if i in suppress_indices:
                continue

            det_i = detections[i]
            box_i = det_boxes[i]

            for j in range(i + 1, len(detections)):
                if j in suppress_indices:
                    continue

                box_j = det_boxes[j]
                iou = self._compute_iou(box_i, box_j)

                if iou > iou_threshold:
                    # High overlap - keep the larger one (likely the real player)
                    if det_i.area >= detections[j].area:
                        suppress_indices.add(j)
                    else:
                        suppress_indices.add(i)
                        det_i = detections[j]
                        box_i = det_boxes[j]

        for i, det in enumerate(detections):
            if i not in suppress_indices:
                keep_detections.append(det)

        return keep_detections
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
    
    
# need this????? probably can delete if NFLDetectionFilterConfig is sufficient
def quick_filter(detections: List[DetectionResult],
                 min_confidence: float = 0.25,
                 min_area: int = 800,
                 max_area: int = 20000) -> List[DetectionResult]:
    """
    Quick NFL-style detection filter.
    """
    filtered = []

    for det in detections:
        # Confidence check
        if det.confidence < min_confidence:
            continue

        # Area check
        area = det.area
        if area < min_area or area > max_area:
            continue

        # Aspect ratio check (players are taller than wide)
        aspect = det.width / det.height if det.height > 0 else 0
        if aspect < 0.2 or aspect > 0.9:
            continue

        filtered.append(det)

    return filtered