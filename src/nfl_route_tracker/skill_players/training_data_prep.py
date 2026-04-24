"""NFL Route Tracker - Phase 2: Training Data Preparation
=======================================================
Prepares synthetic route dataset for route classification training.

Supports:
- Synthetic route generation with configurable counts per class
- Integration with manually labeled real trajectories
- Mixed dataset (synthetic + labeled real)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .synthetic_route_generator import (
    SyntheticRouteGenerator, SyntheticRoute,
    RouteType, ALL_ROUTE_TYPES, generate_synthetic_dataset
)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingDataConfig:
    """Configuration for training data generation."""

    # Output
    output_dir: Path = Path("data/training_data")
    num_routes_per_type: int = 500  # Recommended minimum for deep learning

    # Route types to generate
    route_types: List[RouteType] = None

    # Per-class counts (overrides num_routes_per_type for specific classes)
    class_counts: Dict[str, int] = None

    # Trajectory normalization
    num_points: int = 64  # Fixed length for all trajectories
    normalize_scale: bool = True

    # Augmentation
    add_noise: bool = True
    add_flips: bool = True

    # Manual labels integration
    include_manual_labels: bool = False
    manual_labels_path: Optional[Path] = None

    random_seed: int = 42  # For reproducibility

    # Class weights (for imbalanced data)
    use_class_weights: bool = False

    def __post_init__(self):
        if self.route_types is None:
            self.route_types = ALL_ROUTE_TYPES.copy()
        if self.class_counts is None:
            self.class_counts = {}


class SyntheticTrainingDataGenerator:
    """
    Generates synthetic route training data for classification models.
    """

    def __init__(self, config: Optional[TrainingDataConfig] = None):
        self.config = config or TrainingDataConfig()
        self.generator = SyntheticRouteGenerator(random_seed=self.config.random_seed)

    def generate(self) -> List[SyntheticRoute]:
        """Generate the complete synthetic dataset."""
        print("=" * 60)
        print("Synthetic Route Training Data Generation")
        print("=" * 60)

        all_routes = []
        route_types = self.config.route_types

        # Generate routes with configurable counts per class
        print(f"\nGenerating routes with per-class counts...")
        print(f"Route types: {[rt.value for rt in route_types]}")

        for rt in route_types:
            # Check for per-class override
            count = self.config.class_counts.get(rt.value, self.config.num_routes_per_type)

            routes = self.generator.generate_dataset(route_types=[rt],
                                                     routes_per_type=count,
                                                     param_variations=True)
            all_routes.extend(routes)
            print(f"  - {rt.value}: {count} routes (total so far: {len(all_routes)})")

        print(f"\nTotal routes generated: {len(all_routes)}")

        # Add augmented versions if configured
        if self.config.add_flips:
            print("\nGenerating Y-flipped augmentations...")
            all_routes = self._add_flip_augmentations(all_routes)
            print(f"  Total after flips: {len(all_routes)}")

        # Load manual labels if configured
        if self.config.include_manual_labels and self.config.manual_labels_path:
            manual_routes = self._load_manual_labels()
            if self.config.add_flips:
                print("\nGenerating Y-flipped augmentations (manual labels)...")
                manual_routes = self._add_flip_augmentations(manual_routes)
            all_routes.extend(manual_routes)
            print(f"\nAdded {len(manual_routes)} manually labeled routes")
            print(f"  Total after manual labels: {len(all_routes)}")

        return all_routes

    def _load_manual_labels(self) -> List[SyntheticRoute]:
        """Load and convert manually labeled trajectories."""
        if not self.config.manual_labels_path:
            return []

        if not Path(self.config.manual_labels_path).exists():
            print(f"  Warning: Manual labels file not found: {self.config.manual_labels_path}")
            return []

        with open(self.config.manual_labels_path, 'r') as f:
            data = json.load(f)

        routes = []
        for label in data.get('labels', []):
            try:
                route_type = RouteType(label['route_type'])
                route = SyntheticRoute(route_type=route_type,
                                       x_coords=np.array(label['x_coords']),
                                       y_coords=np.array(label['y_coords']),
                                       params=None,  # Manual routes don't have params
                                       num_points=len(label['x_coords']))
                routes.append(route)
            except (KeyError, ValueError):
                # Skip invalid labels
                continue

        print(f"  Loaded {len(routes)} manual labels")
        return routes

    def _add_flip_augmentations(self, routes: List[SyntheticRoute]) -> List[SyntheticRoute]:
        """Add Y-flipped versions of routes as data augmentation."""
        augmented = []

        for route in routes:
            # Create flipped version
            flipped_x = route.x_coords.copy()
            flipped_y = 1.0 - route.y_coords  # Y-flip

            # Create new SyntheticRoute with flipped data
            flipped_route = SyntheticRoute(route_type=route.route_type,
                                           x_coords=flipped_x,
                                           y_coords=flipped_y,
                                           params=route.params,
                                           num_points=route.num_points)
            augmented.append(flipped_route)

        return routes + augmented

    def prepare_for_model(self, routes: List[SyntheticRoute],
                         output_format: str = "json") -> Dict:
        """
        Prepare routes in format suitable for model training.
        """
        print("\nPreparing data for model training...")

        # Get class names
        class_names = sorted(list(set(r.route_type.value for r in routes)))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Convert to arrays
        num_points = self.config.num_points
        X = np.zeros((len(routes), num_points, 2), dtype=np.float32)
        y = np.zeros(len(routes), dtype=np.int32)

        for i, route in enumerate(routes):
            # Ensure fixed length
            x = route.x_coords[:num_points]
            y_coord = route.y_coords[:num_points]

            # Pad if necessary
            if len(x) < num_points:
                x = np.pad(x, (0, num_points - len(x)), mode='edge')
                y_coord = np.pad(y_coord, (0, num_points - len(y_coord)), mode='edge')

            X[i, :, 0] = x
            X[i, :, 1] = y_coord
            y[i] = class_to_idx[route.route_type.value]

        # Save in requested format
        if output_format == "numpy":
            return {'X_train': X,
                    'y_train': y,
                    'class_names': class_names,
                    'num_classes': len(class_names)}
        else:
            # JSON format
            return {'num_classes': len(class_names),
                    'class_names': class_names,
                    'num_samples': len(routes),
                    'num_points': num_points,
                    'routes': [r.to_normalized_dict() for r in routes]}

    def save(self, routes: List[SyntheticRoute]):
        """Save generated training data to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save full JSON dataset
        data = self.prepare_for_model(routes, output_format="json")
        json_path = self.config.output_dir / "synthetic_routes.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved JSON dataset: {json_path}")

        # Save numpy arrays for training
        np_data = self.prepare_for_model(routes, output_format="numpy")
        np.savez(self.config.output_dir / "training_data.npz",
                X_train=np_data['X_train'],
                y_train=np_data['y_train'])
        print(f"Saved numpy arrays: {self.config.output_dir / 'training_data.npz'}")

        # Save class mapping
        class_path = self.config.output_dir / "class_mapping.json"
        with open(class_path, 'w') as f:
            json.dump({
                'class_names': np_data['class_names'],
                'num_classes': np_data['num_classes']
            }, f, indent=2)
        print(f"Saved class mapping: {class_path}")

        # Save summary
        summary = {
            'total_samples': len(routes),
            'num_classes': np_data['num_classes'],
            'class_names': np_data['class_names'],
            'samples_per_class': {
                name: int(np.sum(np_data['y_train'] == idx))
                for name, idx in zip(np_data['class_names'],
                                    range(np_data['num_classes']))
            },
            'num_points': self.config.num_points,
            'random_seed': self.config.random_seed
        }
        summary_path = self.config.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_path}")

    def run(self) -> Dict:
        """Run complete training data generation."""
        routes = self.generate()
        self.save(routes)

        # Print class distribution
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        for rt in self.config.route_types:
            count = sum(1 for r in routes if r.route_type == rt)
            print(f"  {rt.value}: {count}")
        print("=" * 60)

        return {'num_routes': len(routes)}

def generate_training_data(output_dir: str = "data/training_data",
                            num_routes_per_type: int = 500,
                            random_seed: int = 42,
                            add_flips: bool = True,
                            include_manual_labels: bool = False,
                            manual_labels_path: Optional[str] = None,) -> int:
    """
    Convenience function to generate training data.
    """
    config = TrainingDataConfig(output_dir=Path(output_dir),
                                num_routes_per_type=num_routes_per_type,
                                class_counts = {'streak': 600, 'slant': 500,  'out':500,
                                                'flat': 500, 'curl': 600, 'dig': 500,
                                                'corner': 300, 'post': 400, 'drag': 300,
                                                'comeback': 200, 'wheel': 300},
                                random_seed=random_seed,
                                add_flips=add_flips,
                                include_manual_labels=include_manual_labels,
                                manual_labels_path=Path(manual_labels_path) if manual_labels_path else None,)

    gen = SyntheticTrainingDataGenerator(config)
    result = gen.run()
    return result['num_routes']


# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic route training data")
    parser.add_argument("--output-dir", default="data/training_data", help="Output directory for training data")
    parser.add_argument("--num-routes", type=int, default=500, help="Number of routes per type to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-flips", action="store_true", help="Don't add Y-flip augmentations")
    parser.add_argument("--include-manual-labels", action="store_true",  help="Include manually labeled trajectories from --manual-labels-path")
    parser.add_argument("--manual-labels-path", type=Path,  default=Path("data/manual_labels/training_labels.json"),  help="Path to manual labels JSON file")

    args = parser.parse_args()

    # Create config with manual labels integration
    config = TrainingDataConfig(output_dir=Path(args.output_dir),
                                num_routes_per_type=args.num_routes,
                                random_seed=args.seed,
                                class_counts = {'streak': 600, 'slant': 500,  'out':500,
                                                'flat': 500, 'curl': 600, 'dig': 500,
                                                'corner': 300, 'post': 400, 'drag': 300,
                                                'comeback': 200, 'wheel': 300},
                                add_flips=not args.no_flips,
                                include_manual_labels=args.include_manual_labels,
                                manual_labels_path=args.manual_labels_path if args.include_manual_labels else None)

    gen = SyntheticTrainingDataGenerator(config)
    gen.run()

if __name__ == "__main__":
    main()
