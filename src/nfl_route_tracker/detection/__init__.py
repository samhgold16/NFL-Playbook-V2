"""
NFL Route Tracker - Detection Module
=====================================

Object detection using YOLO for identifying players in video frames.
"""

from .player_detector import PlayerDetector, DetectionResult

__all__ = [
    'PlayerDetector',
    'DetectionResult'
]
