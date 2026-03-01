"""
NFL Route Tracker - Object Tracker Module
==========================================

Multi-object tracking using DeepSORT algorithm, combining with YOLO algorithm.
This maintains consistent track IDs across frames even through occlusions.
"""

# important packages
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

# importing global variables and other functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.detection.player_detector import DetectionResult
from nfl_route_tracker.tracking.trajectory import Detection, TrajectoryStore