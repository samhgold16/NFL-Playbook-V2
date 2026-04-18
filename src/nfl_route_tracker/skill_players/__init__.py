"""NFL Route Tracker - Phase 2: Route Classification
=================================================
This module contains components for:
- Loading and preprocessing Phase 1 trajectory data
- Classifying players as offensive/defensive
- Filtering skill position players (WR, TE, RB)
- Generating synthetic route data for training
"""

from .data_loader import (
    TrajectoryDataLoader,
    load_trajectory_from_json,
    load_all_trajectories_from_directory,
)
from .offense_defense_classifier import (
    LineOfScrimmage,
    LineOfScrimmageClassifier,
    classify_offense_defense,
    fit_line_of_scrimmage,
)
from .skill_position_filter import (
    SkillPositionFilter,
    filter_skill_position_players,
    PlayerClassification,
)
# from .trajectory_preprocessor import (
#     TrajectoryPreprocessor,
#     normalize_trajectory,
#     resample_trajectory,
#     extract_trajectory_features,
# )
# from .synthetic_route_generator import (
#     SyntheticRouteGenerator,
#     RouteType,
#     generate_synthetic_dataset,
#     STREAK_ROUTES,
#     SLANT_ROUTES,
#     POST_ROUTES,
#     CORNER_ROUTES,
#     DRAG_ROUTES,
#     HITCH_ROUTES,
#     IN_ROUTES,
#     DIG_ROUTES,
#     COMEBACK_ROUTES,
#     FLAT_ROUTES,
#     WHEEL_ROUTES,
#     ALL_ROUTE_TYPES,
# )

__all__ = [
    # Data loading
    'TrajectoryDataLoader',
    'load_trajectory_from_json',
    'load_all_trajectories_from_directory',

    # Offense/Defense classification
    'LineOfScrimmage',
    'LineOfScrimmageClassifier',
    'classify_offense_defense',
    'fit_line_of_scrimmage',

    # Skill position filtering
    'SkillPositionFilter',
    'filter_skill_position_players',
    'PlayerClassification',

    # Trajectory preprocessing
    # 'TrajectoryPreprocessor',
    # 'normalize_trajectory',
    # 'resample_trajectory',
    # 'extract_trajectory_features',

    # # Synthetic route generation
    # 'SyntheticRouteGenerator',
    # 'RouteType',
    # 'generate_synthetic_dataset',
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
