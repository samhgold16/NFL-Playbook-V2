"""NFL Route Tracker - Phase 2: Route Classification
=================================================
This module contains components for:
- Loading and preprocessing Phase 1 trajectory data
- Classifying players as offensive/defensive
- Filtering skill position players (WR, TE, RB)
- Generating synthetic route data for training
- Batch extraction pipeline for processing multiple videos
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
from .synthetic_route_generator import (
    SyntheticRouteGenerator,
    RouteType,
    generate_synthetic_dataset,
    STREAK_ROUTES,
    SLANT_ROUTES,
    POST_ROUTES,
    CORNER_ROUTES,
    DRAG_ROUTES,
    CURL_ROUTES,
    DIG_ROUTES,
    OUT_ROUTES,
    COMEBACK_ROUTES,
    FLAT_ROUTES,
    WHEEL_ROUTES,
    ALL_ROUTE_TYPES,
)
from .pipeline import (
    SkillPositionExtractionPipeline,
    PipelineConfig,
    ExtractionResult,
    run_pipeline,
)
from .training_data_prep import (
    SyntheticTrainingDataGenerator,
    TrainingDataConfig,
    generate_training_data,
)
# from .model_architectures import (
#     RouteClassificationCNN,
#     RouteClassificationCNNResidual,
#     RouteClassificationTransformer,
#     RouteClassificationLSTM,
#     RouteClassificationGRU,
#     get_model,
#     count_parameters,
#     MODEL_REGISTRY,
# )
# from .train_route_classifier import (
#     train_model,
#     load_training_data,
#     prepare_data_for_model,
# )
# from .inference import (
#     RouteClassifier,
#     preprocess_trajectory,
#     preprocess_trajectory_from_coords,
#     classify_skill_positions,
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

    # Synthetic route generation
    'SyntheticRouteGenerator',
    'RouteType',
    'generate_synthetic_dataset',
    'STREAK_ROUTES',
    'SLANT_ROUTES',
    'POST_ROUTES',
    'CORNER_ROUTES',
    'DRAG_ROUTES',
    'CURL_ROUTES',
    'DIG_ROUTES',
    'OUT_ROUTES',
    'COMEBACK_ROUTES',
    'FLAT_ROUTES',
    'WHEEL_ROUTES',
    'ALL_ROUTE_TYPES',

    # Pipeline
    'SkillPositionExtractionPipeline',
    'PipelineConfig',
    'ExtractionResult',
    'run_pipeline',

    # Training data preparation
    'SyntheticTrainingDataGenerator',
    'TrainingDataConfig',
    'generate_training_data',

    # Model architectures
    # 'RouteClassificationCNN',
    # 'RouteClassificationCNNResidual',
    # 'RouteClassificationTransformer',
    # 'RouteClassificationLSTM',
    # 'RouteClassificationGRU',
    # 'get_model',
    # 'count_parameters',
    # 'MODEL_REGISTRY',

    # # Training
    # 'train_model',
    # 'load_training_data',
    # 'prepare_data_for_model',

    # # Inference
    # 'RouteClassifier',
    # 'preprocess_trajectory',
    # 'preprocess_trajectory_from_coords',
    # 'classify_skill_positions',
]
