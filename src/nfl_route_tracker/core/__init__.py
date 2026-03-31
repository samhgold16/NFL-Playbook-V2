"""
NFL Route Tracker - Core Module
================================

Core utilities for video loading and configuration.
"""

from .config import (
    NFLFieldConstants,
    NFL_FIELD
)
from .video_loader import VideoLoader, VideoMetadata

__all__ = [
    'NFLFieldConstants',
    'NFL_FIELD',
    'VideoLoader',
    'VideoMetadata'
]
