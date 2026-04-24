"""
NFL Route Tracker - Image Data Preparation
==========================================
Converts synthetic routes and manual labels into a .npz image dataset
for image-based CNN classification in Google Colab.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw


# =============================================================================
# Trajectory → Image
# =============================================================================

def trajectory_to_image(x_coords: np.ndarray, y_coords: np.ndarray,
                        size: int = 64, line_color: Tuple[int, int, int] = (30, 100, 255),
                        line_width: int = 2, start_color: Tuple[int, int, int] = (0, 210, 0),
                        end_color: Tuple[int, int, int] = (220, 30, 30), dot_radius: int = 3,
                        padding: int = 5) -> np.ndarray:
    """
    Render a single trajectory as an RGB image.
    """
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Scale normalized [0,1] coords to pixel space with padding
    drawable = size - 2 * padding
    px = (np.array(x_coords, dtype=np.float32) * drawable + padding).astype(int)
    py = (np.array(y_coords, dtype=np.float32) * drawable + padding).astype(int)

    # Clip to valid pixel range
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)

    points = [(int(x), int(y)) for x, y in zip(px, py)]

    # Draw route line
    if len(points) >= 2:
        draw.line(points, fill=line_color, width=line_width)

    # Start marker (green)
    sx, sy = points[0]
    draw.ellipse([sx - dot_radius, sy - dot_radius,
                  sx + dot_radius, sy + dot_radius], fill=start_color)

    # End marker (red)
    ex, ey = points[-1]
    draw.ellipse([ex - dot_radius, ey - dot_radius,
                  ex + dot_radius, ey + dot_radius], fill=end_color)

    # Convert PIL → numpy (H, W, C) → (C, H, W) for PyTorch
    arr = np.array(img, dtype=np.uint8)          # (H, W, 3)
    arr = arr.transpose(2, 0, 1)                  # (3, H, W)
    return arr


# =============================================================================
# Data loaders
# =============================================================================

def load_synthetic_routes(path: Path) -> List[Dict]:
    """
    Load from data/training_data/synthetic_routes.json.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    routes = []
    for r in data.get('routes', []):
        routes.append({'x_coords':   np.array(r['x_coords'],  dtype=np.float32),
                        'y_coords':   np.array(r['y_coords'],  dtype=np.float32),
                        'route_type': r['route_type'],
                        'source':     'synthetic',})
    print(f"  Loaded {len(routes)} synthetic routes from {path.name}")
    return routes


def load_manual_labels(path: Path) -> List[Dict]:
    """
    Load from data/manual_labels/training_labels.json.
    """
    if not path.exists():
        print(f"  Manual labels not found at {path} — skipping.")
        return []

    with open(path, 'r') as f:
        data = json.load(f)

    routes = []
    for item in data.get('labels', []):
        if 'x_coords' not in item or 'y_coords' not in item:
            continue  # labels.json entries without coords are skipped
        routes.append({'x_coords':   np.array(item['x_coords'],  dtype=np.float32),
                        'y_coords':   np.array(item['y_coords'],  dtype=np.float32),
                        'route_type': item['route_type'],
                        'source':     'manual', })
    print(f"  Loaded {len(routes)} manual label routes from {path.name}")
    return routes


# =============================================================================
# Class mapping
# =============================================================================

def build_class_mapping(routes: List[Dict], existing_mapping_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Build class_name to index mapping.
    """
    if existing_mapping_path and existing_mapping_path.exists():
        with open(existing_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        class_names = mapping_data['class_names']
        print(f"  Reusing existing class mapping: {class_names}")
    else:
        class_names = sorted(set(r['route_type'] for r in routes))
        print(f"  Built new class mapping: {class_names}")

    return {name: idx for idx, name in enumerate(class_names)}, class_names


# =============================================================================
# Main conversion
# =============================================================================

def convert_to_image_dataset(synthetic_path: Path,
                            manual_labels_path: Path,
                            output_dir: Path,
                            class_mapping_path: Path,
                            image_size: int = 64,
                            save_preview: bool = False,) -> None:
    """
    Full pipeline: load trajectories to render images to save .npz
    """
    print("\n" + "=" * 60)
    print("NFL Route Image Dataset Generator")
    print("=" * 60)

    # load all routes
    print("\nLoading route data...")
    routes = []
    routes.extend(load_synthetic_routes(synthetic_path))
    routes.extend(load_manual_labels(manual_labels_path))
    print(f"  Total routes: {len(routes)}")

    # class mapping
    print("\nBuilding class mapping...")
    class_to_idx, class_names = build_class_mapping(routes, class_mapping_path)
    num_classes = len(class_names)

    # filter bad labels
    before = len(routes)
    routes = [r for r in routes if r['route_type'] in class_to_idx]
    if len(routes) < before:
        print(f"  Dropped {before - len(routes)} routes with unknown labels")

    # redner images
    print(f"\nRendering {len(routes)} images at {image_size}x{image_size}...")
    N = len(routes)
    X = np.zeros((N, 3, image_size, image_size), dtype=np.uint8)
    y = np.zeros(N, dtype=np.int32)

    for i, route in enumerate(routes):
        X[i] = trajectory_to_image(route['x_coords'], route['y_coords'], size=image_size,)
        y[i] = class_to_idx[route['route_type']]

        if (i + 1) % 1000 == 0:
            print(f"  Rendered {i + 1}/{N}...")

    print(f"  Done. X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Memory: {X.nbytes / 1e6:.1f} MB")

    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / 'image_training_data.npz'
    np.savez_compressed(npz_path, X_train=X, y_train=y)
    print(f"\nSaved: {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    # print summary stats
    samples_per_class = {name: int(np.sum(y == idx)) for name, idx in class_to_idx.items()}
    summary = {'image_size': image_size,
               'num_samples': int(N),
               'num_classes': num_classes,
               'class_names': class_names,
               'samples_per_class': samples_per_class,
               'sources': {'synthetic': sum(1 for r in routes if r['source'] == 'synthetic'),
                           'manual': sum(1 for r in routes if r['source'] == 'manual'),},
               'array_shape':  list(X.shape),
               'array_dtype':  str(X.dtype)}
    summary_path = output_dir / 'image_dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # view grid, used for presentation
    if save_preview:
        _save_preview_grid(X, y, class_names, output_dir, image_size)


def _save_preview_grid(X: np.ndarray, y: np.ndarray,
                      class_names: List[str], output_dir: Path,
                      image_size: int, samples_per_class: int = 4) -> None:
    """
    Save a grid image showing sample routes per class.
    """
    print("\nGenerating preview grid...")
    n_classes = len(class_names)
    cols = samples_per_class
    rows = n_classes

    border = 2
    label_width = 80
    cell = image_size + border
    grid_w = label_width + cols * cell + border
    grid_h = rows * cell + border

    grid = Image.new('RGB', (grid_w, grid_h), (220, 220, 220))

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw = Image.new('RGB', (label_width - 4, image_size), (240, 240, 240))

    for class_idx, class_name in enumerate(class_names):
        # Pick up to `samples_per_class` samples for this class
        indices = np.where(y == class_idx)[0][:samples_per_class]

        for col_idx, sample_idx in enumerate(indices):
            img_arr = X[sample_idx].transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
            img_pil = Image.fromarray(img_arr, mode='RGB')

            paste_x = label_width + col_idx * cell + border
            paste_y = class_idx * cell + border
            grid.paste(img_pil, (paste_x, paste_y))

        # Class label on the left
        label_img = Image.new('RGB', (label_width - 4, image_size), (245, 245, 245))
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((4, image_size // 2 - 6), class_name,
                        fill=(30, 30, 30), font=font)
        grid.paste(label_img, (2, class_idx * cell + border))

    preview_path = output_dir / 'route_image_preview.png'
    grid.save(preview_path)
    print(f"Saved preview grid: {preview_path}")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert NFL route trajectories to image dataset for CNN training")
    parser.add_argument('--synthetic-path', type=Path, default=Path('data/training_data/synthetic_routes.json'), help='Path to synthetic_routes.json')
    parser.add_argument( '--manual-labels-path', type=Path, default=Path('data/manual_labels/training_labels.json'), help='Path to training_labels.json (manual labels with coords)')
    parser.add_argument('--output-dir', type=Path,  default=Path('data/training_data'), help='Output directory (default: data/training_data)')
    parser.add_argument('--class-mapping-path', type=Path, default=Path('data/training_data/class_mapping.json'), help='Existing class_mapping.json to keep indices consistent')
    parser.add_argument('--preview', action='store_true', help='Save a sample grid image for visual verification')
    args = parser.parse_args()

    convert_to_image_dataset(synthetic_path=args.synthetic_path,
                            manual_labels_path=args.manual_labels_path,
                            output_dir=args.output_dir,
                            class_mapping_path=args.class_mapping_path,
                            image_size= 64,
                            save_preview=args.preview)

if __name__ == '__main__':
    main()