"""
NFL Route Tracker - Player Detector Module
===========================================

This module uses YOLO (You Only Look Once) for detecting players in video frames, a more complex attempt than temporal derivative in version1
"""

# important packages
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

# importing global variables and other functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_route_tracker.core.config import DetectorConfig
from nfl_route_tracker.core.video_loader import VideoLoader

# similar to MotionBlob in motion_tracker.py
@dataclass
class DetectionResult:
    """
    A single detection from YOLO - similar to the MotionBlob in motion_tracker V1
    Holds attributes for bounding box.
    """
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of detection."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height,
                'confidence': self.confidence,
                'class_id': self.class_id,
                'class_name': self.class_name}
        
# main class
class PlayerDetector:
    """
    Detects players in video frames using YOLOv8.

    This class wraps the Ultralytics YOLO implementation and provides
    a clean interface for detecting people in NFL footage.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the player detector based on default or specified
        """
        self.config = config or DetectorConfig()

        # import YOLO here to make it optional dependency
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("YOLOv8 not installed. Install with: pip install ultralytics")

        print(f"\nLoading YOLO model: {self.config.model_name}")

        # Determine device
        device = self.config.device
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        print(f"Using device: {device}")

        # Load model
        self.model = YOLO(self.config.model_name)
        self.model.to(device)
        self.device = device

        # Track statistics
        self._total_detections = 0
        self._frames_processed = 0

    # for detecting a single frame
    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[DetectionResult]:
        """
        Detect players in a single frame.
        """
        # Run the actual YOLO model using specified parameters
        results = self.model(frame, conf = self.config.confidence_threshold, 
                             classes = self.config.classes, imgsz = self.config.imgsz, 
                             verbose = False)

        # Parse results
        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get bounding box (xyxy format), top-left x, top-lfet y, bottomright x, bottommrighty
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get class name
                cls_name = self.model.names[cls_id]

                # assign all attributes based on yolo results and store
                detection = DetectionResult(x = float(x1), y = float(y1),
                                            width = float(x2 - x1), height = float(y2 - y1),
                                            confidence = conf, class_id = cls_id, class_name = cls_name)

                detections.append(detection)

        # update statistics
        self._frames_processed += 1
        self._total_detections += len(detections)

        if verbose and detections:
            print(f"Detected {len(detections)} players")

        return detections
    
    # now take detect() and do it overall all frames
    def detect_batch(self, frames: List[np.ndarray], verbose: bool = False) -> List[List[DetectionResult]]:
        """
        Detect players in multiple frames (batch processing).
        """

        # Run batch inference
        results = self.model(frames, conf = self.config.confidence_threshold,
                             classes = self.config.classes, imgsz = self.config.imgsz,
                             verbose = False)

        # Parse results for each frame
        all_detections = []

        # same iterative process as single detect() function
        for result in results:
            frame_detections = []
            boxes = result.boxes

            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.model.names[cls_id]

                    detection = DetectionResult(x = float(x1), y = float(y1),
                                                width = float(x2 - x1), height = float(y2 - y1),
                                                confidence = conf, class_id = cls_id, class_name = cls_name)
                    frame_detections.append(detection)
            
            # storing
            all_detections.append(frame_detections)
            self._frames_processed += 1
            self._total_detections += len(frame_detections)

        if verbose:
            total = sum(len(d) for d in all_detections)
            print(f"Total detections: {total}")

        return all_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult],
                        color: Tuple[int, int, int] = (180, 50, 60), thickness: int = 4,
                        show_confidence: bool = True, show_label: bool = True) -> np.ndarray:
        """
        Draw detection/bounding boxes on an individual frame.
        """
        output = frame.copy()

        for det in detections:
            # Draw bounding box
            x1, y1 = int(det.x), int(det.y)
            x2, y2 = int(det.x + det.width), int(det.y + det.height)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw center point
            cx, cy = det.center
            cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # Build label text
            label_parts = []
            if show_label:
                label_parts.append(det.class_name)
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)

                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness)

                # Draw background rectangle
                cv2.rectangle(output, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1),
                              color, -1)

                # Draw text
                cv2.putText(output, label, (x1 + 2, y1 - 5),
                    font, font_scale, (0, 0, 0), thickness)

        return output

    # simple getter/setters for overall testing about how peocessing worked
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {'frames_processed': self._frames_processed,
                'total_detections': self._total_detections,
                'avg_detections_per_frame': (self._total_detections / self._frames_processed if self._frames_processed > 0 else 0)}

    def reset_statistics(self):
        """Reset detection statistics."""
        self._total_detections = 0
        self._frames_processed = 0

#########################################

# Convenience function for quick detection
def detect_players_in_frame(frame: np.ndarray, confidence: float = 0.25, model: str = 'yolov8n.pt') -> List[DetectionResult]:
    """
    Quick function to detect players in a single frame.

    Creates a detector, runs detection, and returns results.
    For processing multiple frames, use PlayerDetector class directly.
    """
    config = DetectorConfig(model_name = model, confidence_threshold = confidence)
    detector = PlayerDetector(config)

    return detector.detect(frame)