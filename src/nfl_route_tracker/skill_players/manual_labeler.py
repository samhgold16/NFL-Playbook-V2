"""
NFL Route Tracker - Phase 2: Manual Labeling Tool
==================================================
Tool for manually labeling extracted trajectories with route types.

Needed LLM help to set up a process to seamlessly manually label trajectories from JSON file
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

# For visualization
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =============================================================================
# Route Types
# =============================================================================

ROUTE_TYPES = ['streak', 'slant', 'post', 'corner', 'drag',
               'curl', 'dig', 'out', 'comeback', 'flat', 'wheel']

# =============================================================================
# Label Dataclasses
# =============================================================================

@dataclass
class TrajectoryLabel:
    """A single labeled trajectory."""
    track_id: int
    source_video: str
    route_type: str
    labeler: str = "manual"
    confidence: int = 5  # 1-5 scale
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'source_video': self.source_video,
            'route_type': self.route_type,
            'labeler': self.labeler,
            'confidence': self.confidence,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectoryLabel':
        return cls(
            track_id = data['track_id'],
            source_video = data['source_video'],
            route_type = data['route_type'],
            labeler = data.get('labeler', 'manual'),
            confidence = data.get('confidence', 5),
            notes = data.get('notes', ''),
        )


@dataclass
class LabelDataset:
    """Collection of labeled trajectories."""
    labels: List[TrajectoryLabel] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_label(self, label: TrajectoryLabel):
        self.labels.append(label)

    def get_label(self, track_id: int, source_video: str) -> Optional[TrajectoryLabel]:
        for label in self.labels:
            if label.track_id == track_id and label.source_video == source_video:
                return label
        return None

    def get_labels_by_type(self, route_type: str) -> List[TrajectoryLabel]:
        return [l for l in self.labels if l.route_type == route_type]

    def to_dict(self) -> Dict:
        return {'metadata': {'total_labels': len(self.labels), 
                             'classes': list(set(l.route_type for l in self.labels)), 
                             **self.metadata},
                'labels': [l.to_dict() for l in self.labels]}

    @classmethod
    def from_dict(cls, data: Dict) -> 'LabelDataset':
        dataset = cls()
        dataset.metadata = data.get('metadata', {})
        dataset.labels = [TrajectoryLabel.from_dict(l) for l in data.get('labels', [])]
        return dataset

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'LabelDataset':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Interactive Labeler
# =============================================================================

class ManualLabeler:
    """
    Manual labeling command line tool for trajectories.
    """

    def __init__(self, trajectories_path: Path, output_dir: Path, existing_labels: Optional[Path] = None):
        self.trajectories_path = trajectories_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
        self.trajectories = data.get('trajectories', [])
        self.num_points = data.get('num_points', 64)

        # Load or create labels
        self.labels_file = self.output_dir / "labels.json"
        if existing_labels and existing_labels.exists():
            self.labels = LabelDataset.load(existing_labels)
            self.labeled_ids = set(
                (l.track_id, l.source_video) for l in self.labels.labels
            )
        elif self.labels_file.exists():
            self.labels = LabelDataset.load(self.labels_file)
            self.labeled_ids = set(
                (l.track_id, l.source_video) for l in self.labels.labels
            )
        else:
            self.labels = LabelDataset()
            self.labeled_ids = set()

    def get_unlabeled_trajectories(self) -> List[Dict]:
        """Get trajectories that haven't been labeled yet."""
        unlabeled = []
        for traj in self.trajectories:
            key = (traj.get('track_id', 0), traj.get('source_video', ''))
            if key not in self.labeled_ids:
                unlabeled.append(traj)
        return unlabeled

    def label_trajectories(self, batch_size: int = 20, strategy: str = "random"):
        """
        Interactive labeling session.
        """
        unlabeled = self.get_unlabeled_trajectories()

        if not unlabeled:
            print("All trajectories have been labeled!")
            return

        print(f"\n{'='*60}")
        print("NFL Route Tracker - Manual Labeling Tool")
        print(f"{'='*60}")
        print(f"Total trajectories: {len(self.trajectories)}")
        print(f"Already labeled:   {len(self.labeled_ids)}")
        print(f"Remaining:        {len(unlabeled)}")
        print(f"Batch size:       {batch_size}")
        print(f"Strategy:          {strategy}")
        print(f"{'='*60}")
        print()

        # Select trajectories to label
        if strategy == "random":
            batch = random.sample(unlabeled, min(batch_size, len(unlabeled)))
        else:
            batch = unlabeled[:batch_size]

        print("\nRoute Types:")
        for i, rt in enumerate(ROUTE_TYPES):
            print(f"  {i+1}: {rt:10s}")
        print(f"  0: Skip this trajectory")
        print(f"  s: Save and exit")
        print(f"  q: Quit without saving")
        print()

        for i, traj in enumerate(batch):
            self._label_single_trajectory(traj, i + 1, len(batch))

        # Save labels
        self.labels.save(self.labels_file)
        print(f"\nLabels saved to: {self.labels_file}")

    # main mechanic, manually entering a number corresponding to route label
    def _label_single_trajectory(self, traj: Dict, index: int, total: int):
        """Label a single trajectory."""
        track_id = traj.get('track_id', 0)
        source = traj.get('source_video', 'unknown')
        x_coords = np.array(traj['x_coords'])
        y_coords = np.array(traj['y_coords'])

        print(f"\n[{index}/{total}] Track ID: {track_id} | Source: {source}")
        print(f"  Points: {len(traj.get('detections', traj.get('x_coords', [])))}")
        print(f"  First: ({x_coords[0]:.2f}, {y_coords[0]:.2f})")
        print(f"  Last:  ({x_coords[-1]:.2f}, {y_coords[-1]:.2f})")

        # Get user input
        try:
            choice = input("  Select route type (1-11): ").strip().lower()

            if choice == 's':
                self.labels.save(self.labels_file)
                print("Labels saved. Exiting.")
                return None
            elif choice == 'q':
                print("Exiting without saving.")
                return None
            elif choice == '0':
                return None

            route_num = int(choice) - 1
            if 0 <= route_num < len(ROUTE_TYPES):
                route_type = ROUTE_TYPES[route_num]

                label = TrajectoryLabel(track_id=track_id, source_video=source,
                                        route_type=route_type, confidence=5, notes="")
                self.labels.add_label(label)
                self.labeled_ids.add((track_id, source))
                print(f"  -> Labeled as: {route_type}")
                return label

        except (ValueError, KeyboardInterrupt):
            print("  Skipped.")
            return None

    def export_labels_for_training(self, holdout_fraction: float = 0.0, holdout_path: Optional[Path] = None,) -> Dict:
        """
        Export labels in format for training_data_prep.
        """
        # Build full label list with coords attached, same setup as orginal JSON files
        label_list = []
        for label in self.labels.labels:
            for traj in self.trajectories:
                if (traj.get('track_id') == label.track_id and
                    traj.get('source_video') == label.source_video):
                    label_list.append({
                        'track_id':    label.track_id,
                        'source_video': label.source_video,
                        'route_type':  label.route_type,
                        'x_coords':    traj['x_coords'],
                        'y_coords':    traj['y_coords'],
                        'confidence':  label.confidence,
                    })
                    break

        if holdout_fraction > 0.0:
            # Stratify by route type so every class is represented in both splits
            from collections import defaultdict
            import math

            by_type = defaultdict(list)
            for item in label_list:
                by_type[item['route_type']].append(item)

            train_list   = []
            holdout_list = []

            for route_type, items in by_type.items():
                random.shuffle(items)
                n_holdout = max(1, math.ceil(len(items) * holdout_fraction))
                # Only hold out if we have enough samples to spare
                if len(items) > 1:
                    holdout_list.extend(items[:n_holdout])
                    train_list.extend(items[n_holdout:])
                else:
                    # Only 1 sample for this class — put in training, warn
                    train_list.extend(items)
                    print(f"  Warning: only 1 label for '{route_type}' — kept in training")

            print(f"\nSplit: {len(train_list)} training / {len(holdout_list)} holdout")

            # Save holdout
            if holdout_path is None:
                holdout_path = self.output_dir / "holdout_labels.json"
            holdout_output = {'num_labels': len(holdout_list),
                              'split': 'holdout',
                              'labels': holdout_list}
            with open(holdout_path, 'w') as f:
                json.dump(holdout_output, f, indent=2)
            print(f"Holdout labels saved to: {holdout_path}")

        else:
            train_list = label_list

        # Save training export
        output = {'num_labels': len(train_list),
                 'split': 'training',
                'labels':  train_list,}
        export_path = self.output_dir / "training_labels.json"
        with open(export_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Training labels saved to: {export_path}  ({len(train_list)} labels)")

        return output

# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Manual labeling tool for trajectory route classification",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input", type=Path, required=True, help="Path to normalized trajectories JSON")
    parser.add_argument("--output", type=Path, default=Path("data/manual_labels"),help="Output directory for labels")
    parser.add_argument("--existing-labels", type=Path, default=None, help="Path to existing labels JSON")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of trajectories to label per batch")
    parser.add_argument("--strategy", choices=["random", "sequential"], default="random", help="Selection strategy for trajectories to label")
    parser.add_argument( "--export", action="store_true", help="Export labels for training")
    parser.add_argument( "--holdout-fraction", type=float, default=0.2, help="Fraction of labels to reserve for evaluation (e.g. 0.2 = 20%). ")

    args = parser.parse_args()

    # Initialize labeler
    labeler = ManualLabeler(trajectories_path=args.input, output_dir=args.output, existing_labels=args.existing_labels)

    if args.export:
        labeler.export_labels_for_training(holdout_fraction=args.holdout_fraction,)
    else:
        labeler.label_trajectories(batch_size=args.batch_size, strategy=args.strategy)

if __name__ == "__main__":
    main()