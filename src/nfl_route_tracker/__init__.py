"""NFL Route Tracker
=================
A Python package for extracting player tracking data from NFL All-22 film using computer vision techniques.

Phase 1: Player Detection and Tracking
Phase 2: Route Classification (Skill Position Identification)
"""

__version__ = "0.5.0"
__author__ = "Sam Gold"

# Core components
from .core import (
    NFLFieldConstants,
    NFL_FIELD,
    VideoLoader,
    VideoMetadata,
)

# Detection components
from .detection import (
    PlayerDetector,
    DetectionResult,
)

# Tracking components
from .tracking import (
    # Data structures
    Detection,
    Trajectory,
    TrajectoryStore,
    # Trackers
    ByteTrackTracker,
    TrackerConfig,
    Track,
    DetectionTracker,
    DetectionTrackerConfig,
    # Field orientation (perspective correction)
    #FieldOrientationDetector,
    #FieldOrientation,
    #FieldOrientationConfig,
    # Trajectory processing
    TrajectoryMerger,
    merge_trajectory_store,
)

# Visualization components
from .visualizations import (
    TrajectoryVisualizer,
    create_tracking_video,
)

# Phase 2: Route Classification components
from .skill_players import (
    # Data loading
    TrajectoryDataLoader,
    load_trajectory_from_json,
    load_all_trajectories_from_directory,
    # Offense/Defense classification
    LineOfScrimmage,
    LineOfScrimmageClassifier,
    classify_offense_defense,
    fit_line_of_scrimmage,
    # Skill position filtering
    SkillPositionFilter,
    filter_skill_position_players,
    PlayerClassification,
    # Trajectory preprocessing
    # TrajectoryPreprocessor,
    # normalize_trajectory,
    # resample_trajectory,
    # extract_trajectory_features,
    # # Synthetic route generation
    # SyntheticRouteGenerator,
    # RouteType,
    # generate_synthetic_dataset,
    # # Route types
    # STREAK_ROUTES,
    # SLANT_ROUTES,
    # POST_ROUTES,
    # CORNER_ROUTES,
    # DRAG_ROUTES,
    # HITCH_ROUTES,
    # IN_ROUTES,
    # DIG_ROUTES,
    # COMEBACK_ROUTES,
    # FLAT_ROUTES,
    # WHEEL_ROUTES,
    # ALL_ROUTE_TYPES,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Core
    'NFLFieldConstants',
    'NFL_FIELD',
    'VideoLoader',
    'VideoMetadata',

    # Detection
    'PlayerDetector',
    'DetectionResult',

    # Tracking - Data structures
    'Detection',
    'Trajectory',
    'TrajectoryStore',
    'ByteTrackTracker',
    'TrackerConfig',
    'Track',
    'DetectionTracker',
    'DetectionTrackerConfig',

    # Field orientation (perspective correction)
    #'FieldOrientationDetector',
    #'FieldOrientation',
    #'FieldOrientationConfig',

    # Trajectory processing
    'TrajectoryMerger',
    'merge_trajectory_store',

    # Visualization
    'TrajectoryVisualizer',
    'create_tracking_video',

    # Phase 2 - Data loading
    'TrajectoryDataLoader',
    'load_trajectory_from_json',
    'load_all_trajectories_from_directory',

    # Phase 2 - Offense/Defense classification
    'LineOfScrimmage',
    'LineOfScrimmageClassifier',
    'classify_offense_defense',
    'fit_line_of_scrimmage',

    # Phase 2 - Skill position filtering
    'SkillPositionFilter',
    'filter_skill_position_players',
    'PlayerClassification',

    # Phase 2 - Trajectory preprocessing
    # 'TrajectoryPreprocessor',
    # 'normalize_trajectory',
    # 'resample_trajectory',
    # 'extract_trajectory_features',

    # # Phase 2 - Synthetic route generation
    # 'SyntheticRouteGenerator',
    # 'RouteType',
    # 'generate_synthetic_dataset',

    # # Phase 2 - Route types
    # 'STREAK_ROUTES',
    # 'SLANT_ROUTES',
    # 'POST_ROUTES',
    # 'CORNER_ROUTES',
    # 'DRAG_ROUTES',
    # 'HITCH_ROUTES',
    # 'IN_ROUTES',
    # 'DIG_ROUTES',
    # 'COMEBACK_ROUTES',
    # 'FLAT_ROUTES',
    # 'WHEEL_ROUTES',
    # 'ALL_ROUTE_TYPES',
]
