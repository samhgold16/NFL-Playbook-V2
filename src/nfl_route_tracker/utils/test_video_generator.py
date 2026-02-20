"""
NFL Route Tracker - Test Video Generator
=========================================

This module creates synthetic test videos with known motion patterns.
These videos are essential for:
1. Unit testing - verifiable ground truth
2. Algorithm development - controlled conditions
3. Debugging - reproducible scenarios

Author: Sam Gold
Phase: 1 - Foundation

Simple Video Types Examples:
---------------------
1. Single object moving linearly
2. Single object with acceleration
3. Multiple objects with different patterns
4. NFL-style route simulations (slant, out, curl, go)
"""

# important packages
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import shutil

# initializing values for video generation
@dataclass
class VideoConfig:
    """Configuration for generated test videos.
    set defaults like dimensions, fps, duration, and background color."""
    width: int = 640
    height: int = 480
    fps: int = 30
    duration_seconds: float = 3.0
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black

    @property
    def total_frames(self) -> int:
        return int(self.fps * self.duration_seconds)

# initializing values for moving objects in video generation
@dataclass
class MovingObject:
    """
    Defines a moving object for test videos.

    Attributes:
    -----------
    start_x, start_y : float
        Initial position
    color : Tuple[int, int, int]
        BGR color
    radius : int
        Size of the object (circle radius)
    trajectory_fn : Callable
        Function that takes (frame_id, total_frames) and returns (x, y)
    """
    start_x: float
    start_y: float
    color: Tuple[int, int, int]
    radius: int
    trajectory_fn: Callable[[int, int], Tuple[float, float]]


class TestVideoGenerator:
    """
    Generates synthetic test videos with known motion patterns.

    This class creates videos where we know exactly where objects should be,
    allowing us to validate our tracking algorithms.

    Example Usage:
    -------------
    ```python
    generator = TestVideoGenerator()

    # Create a simple linear motion video
    generator.create_linear_motion("linear_test.mp4")

    # Create a video with multiple objects
    generator.create_multi_object("multi_test.mp4", num_objects=3)

    # Create NFL-style route videos
    generator.create_route_video("slant_route.mp4", route_type="slant")
    ```
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the generator.

        Parameters:
        -----------
        config : VideoConfig, optional
            Video settings (default: 640x480, 30fps, 3 seconds)
        """
        # use default config, or custom if provided (width, height, fps, duration, background color)
        self.config = config or VideoConfig()
        print(f"Video Generator Object Initialized")
        print(f"Resolution: {self.config.width}x{self.config.height}")
        print(f"FPS: {self.config.fps}")
        print(f"Duration: {self.config.duration_seconds}s")

    def _create_video_writer(self, output_path: str) -> cv2.VideoWriter:
        """Create a video writer with the configured settings."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            output_path,
            fourcc,
            self.config.fps,
            (self.config.width, self.config.height)
        )

    def _create_frame(self) -> np.ndarray:
        """Create an empty frame with the background color."""
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        frame[:] = self.config.background_color
        return frame

    def create_linear_motion(
        self,
        output_path: str,
        direction: str = "right",
        speed: float = 1.0,
        object_color: Tuple[int, int, int] = (255, 255, 255),
        object_radius: int = 20
    ) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Create a video with a single object moving in a straight line.

        This is the simplest test case - one object, constant velocity.

        Parameters:
        -----------
        output_path : str
            Where to save the video
        direction : str
            "right", "left", "up", "down", or "diagonal"
        speed : float
            Speed multiplier (1.0 = normal)
        object_color : Tuple[int, int, int]
            BGR color of the object
        object_radius : int
            Size of the object

        Returns:
        --------
        Tuple[str, List[Tuple[float, float]]]
            - Path to created video
            - List of ground truth positions (x, y) for each frame
        """
        print(f"\n[TestVideoGenerator] Creating linear motion video...")
        print(f"                     Direction: {direction}, Speed: {speed}x")

        # Define trajectory based on direction
        w, h = self.config.width, self.config.height
        total = self.config.total_frames

        if direction == "right":
            def trajectory(frame, total):
                progress = frame / total
                x = object_radius + progress * (w - 2*object_radius) * speed
                y = h // 2
                return min(x, w - object_radius), y
        elif direction == "left":
            def trajectory(frame, total):
                progress = frame / total
                x = (w - object_radius) - progress * (w - 2*object_radius) * speed
                y = h // 2
                return max(x, object_radius), y
        elif direction == "up":
            def trajectory(frame, total):
                progress = frame / total
                x = w // 2
                y = (h - object_radius) - progress * (h - 2*object_radius) * speed
                return x, max(y, object_radius)
        elif direction == "down":
            def trajectory(frame, total):
                progress = frame / total
                x = w // 2
                y = object_radius + progress * (h - 2*object_radius) * speed
                return x, min(y, h - object_radius)
        elif direction == "diagonal":
            def trajectory(frame, total):
                progress = frame / total
                x = object_radius + progress * (w - 2*object_radius) * speed
                y = object_radius + progress * (h - 2*object_radius) * speed
                return min(x, w - object_radius), min(y, h - object_radius)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Generate video
        ground_truth = []
        writer = self._create_video_writer(output_path)

        for frame_id in range(total):
            frame = self._create_frame()

            # Calculate position
            x, y = trajectory(frame_id, total)
            ground_truth.append((x, y))

            # Draw object
            cv2.circle(frame, (int(x), int(y)), object_radius, object_color, -1)

            writer.write(frame)

            if frame_id % 30 == 0:
                print(f"                     Frame {frame_id}/{total}: pos=({x:.1f}, {y:.1f})")

        writer.release()
        print(f"                     Saved to: {output_path}")

        return output_path, ground_truth

    def create_accelerating_motion(
        self,
        output_path: str,
        object_color: Tuple[int, int, int] = (255, 255, 255),
        object_radius: int = 20
    ) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Create a video with an object that accelerates.

        The object starts slow and speeds up over time.
        Useful for testing speed calculations.

        Returns:
        --------
        Tuple[str, List[Tuple[float, float]]]
            - Path to video
            - Ground truth positions
        """
        print(f"\n[TestVideoGenerator] Creating accelerating motion video...")

        w, h = self.config.width, self.config.height
        total = self.config.total_frames

        ground_truth = []
        writer = self._create_video_writer(output_path)

        for frame_id in range(total):
            frame = self._create_frame()

            # Quadratic motion (constant acceleration)
            # x = x0 + v0*t + 0.5*a*t^2
            progress = frame_id / total
            x = object_radius + (progress ** 2) * (w - 2*object_radius)
            y = h // 2

            ground_truth.append((x, y))
            cv2.circle(frame, (int(x), int(y)), object_radius, object_color, -1)

            writer.write(frame)

        writer.release()
        print(f"                     Saved to: {output_path}")

        return output_path, ground_truth

    def create_multi_object(
        self,
        output_path: str,
        num_objects: int = 3,
        crossing: bool = False
    ) -> Tuple[str, List[List[Tuple[float, float]]]]:
        """
        Create a video with multiple moving objects.

        This tests the tracker's ability to maintain identity when
        multiple objects are present.

        Parameters:
        -----------
        output_path : str
            Where to save
        num_objects : int
            Number of objects
        crossing : bool
            If True, objects will cross paths (harder to track)

        Returns:
        --------
        Tuple[str, List[List[Tuple[float, float]]]]
            - Path to video
            - List of ground truth positions for each object
        """
        print(f"\n[TestVideoGenerator] Creating multi-object video...")
        print(f"                     Objects: {num_objects}, Crossing: {crossing}")

        w, h = self.config.width, self.config.height
        total = self.config.total_frames

        # Colors for different objects
        colors = [
            (255, 255, 255),  # White
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
        ]

        # Define trajectories
        ground_truths = [[] for _ in range(num_objects)]
        writer = self._create_video_writer(output_path)

        for frame_id in range(total):
            frame = self._create_frame()
            progress = frame_id / total

            for obj_id in range(num_objects):
                if crossing:
                    # Objects cross in the middle
                    if obj_id % 2 == 0:
                        x = 50 + progress * (w - 100)
                        y = h // 2 - 50 + obj_id * 30
                    else:
                        x = (w - 50) - progress * (w - 100)
                        y = h // 2 - 50 + obj_id * 30
                else:
                    # Parallel motion
                    x = 50 + progress * (w - 100)
                    y = 50 + obj_id * (h - 100) // max(1, num_objects - 1)

                ground_truths[obj_id].append((x, y))

                color = colors[obj_id % len(colors)]
                cv2.circle(frame, (int(x), int(y)), 15, color, -1)

            writer.write(frame)

        writer.release()
        print(f"                     Saved to: {output_path}")

        return output_path, ground_truths

    def create_route_video(
        self,
        output_path: str,
        route_type: str = "out",
        object_color: Tuple[int, int, int] = (255, 255, 255),
        object_radius: int = 15
    ) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Create a video simulating an NFL receiver route.

        This is more realistic - the object runs a pattern that
        resembles an actual football route.

        Available Routes:
        ----------------
        - "go": Straight up the field
        - "slant": Diagonal towards middle
        - "out": Run up, then cut outside
        - "in": Run up, then cut inside (dig)
        - "curl": Run up, then turn back
        - "hitch": Quick run, then stop

        Parameters:
        -----------
        output_path : str
            Where to save
        route_type : str
            Type of route to simulate
        object_color : Tuple
            Object color
        object_radius : int
            Object size

        Returns:
        --------
        Tuple[str, List[Tuple[float, float]]]
            - Path to video
            - Ground truth positions
        """
        print(f"\n[TestVideoGenerator] Creating {route_type.upper()} route video...")

        w, h = self.config.width, self.config.height
        total = self.config.total_frames

        # Starting position (like a WR on the line)
        start_x = w // 4
        start_y = h - 50

        # Define route trajectories
        def go_route(frame, total):
            progress = frame / total
            x = start_x
            y = start_y - progress * (h - 100)
            return x, max(y, 50)

        def slant_route(frame, total):
            progress = frame / total
            x = start_x + progress * 150  # Move towards center
            y = start_y - progress * (h - 100)
            return x, max(y, 50)

        def out_route(frame, total):
            progress = frame / total
            if progress < 0.4:
                # Stem - run straight
                x = start_x
                y = start_y - (progress / 0.4) * (h * 0.4)
            else:
                # Break - cut outside
                break_progress = (progress - 0.4) / 0.6
                x = start_x - break_progress * 150
                y = start_y - h * 0.4
            return max(x, 50), y

        def in_route(frame, total):
            progress = frame / total
            if progress < 0.4:
                x = start_x
                y = start_y - (progress / 0.4) * (h * 0.4)
            else:
                break_progress = (progress - 0.4) / 0.6
                x = start_x + break_progress * 150
                y = start_y - h * 0.4
            return min(x, w - 50), y

        def curl_route(frame, total):
            progress = frame / total
            if progress < 0.5:
                x = start_x
                y = start_y - (progress / 0.5) * (h * 0.4)
            else:
                # Turn back
                break_progress = (progress - 0.5) / 0.5
                x = start_x
                y = start_y - h * 0.4 + break_progress * 50
            return x, y

        def hitch_route(frame, total):
            progress = frame / total
            if progress < 0.3:
                x = start_x
                y = start_y - (progress / 0.3) * (h * 0.2)
            else:
                # Stop
                x = start_x
                y = start_y - h * 0.2
            return x, y

        routes = {
            "go": go_route,
            "slant": slant_route,
            "out": out_route,
            "in": in_route,
            "curl": curl_route,
            "hitch": hitch_route
        }

        if route_type not in routes:
            raise ValueError(f"Unknown route: {route_type}. Available: {list(routes.keys())}")

        trajectory_fn = routes[route_type]

        # Generate video
        ground_truth = []
        writer = self._create_video_writer(output_path)

        for frame_id in range(total):
            frame = self._create_frame()

            # Draw field-like lines (for context)
            for y_line in range(0, h, 50):
                cv2.line(frame, (0, y_line), (w, y_line), (30, 30, 30), 1)

            x, y = trajectory_fn(frame_id, total)
            ground_truth.append((x, y))

            cv2.circle(frame, (int(x), int(y)), object_radius, object_color, -1)

            writer.write(frame)

        writer.release()
        print(f"                     Saved to: {output_path}")
        print(f"                     Total frames: {total}")
        print(f"                     Start: ({start_x}, {start_y})")
        print(f"                     End: ({ground_truth[-1][0]:.1f}, {ground_truth[-1][1]:.1f})")

        return output_path, ground_truth


# # NEED BELOW?????

# # Convenience functions for quick testing
# def create_simple_test_video(output_path: str = "simple_test.mp4") -> str:
#     """Create a simple test video with one object moving right."""
#     gen = TestVideoGenerator()
#     path, _ = gen.create_linear_motion(output_path, direction="right")
#     return path


# def create_route_test_suite(output_dir: str = "test_routes") -> List[str]:
#     """Create a suite of test videos for all route types."""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)

#     gen = TestVideoGenerator()
#     paths = []

#     for route in ["go", "slant", "out", "in", "curl", "hitch"]:
#         path = str(output_dir / f"{route}_route.mp4")
#         gen.create_route_video(path, route_type=route)
#         paths.append(path)

#     return paths


if __name__ == "__main__":
    """
    Test the video generator.
    """
    print("\n" + "="*60)
    print("Testing TestVideoGenerator Module")
    print("="*60 + "\n")

    gen = TestVideoGenerator(VideoConfig(
        width=320,
        height=240,
        fps=30,
        duration_seconds=2.0
    ))

    # mirrors all-22 film normal settings
    gen1 = TestVideoGenerator(VideoConfig(
        width=640,
        height=480,
        fps=24,
        duration_seconds=10.0
    ))

    # Test 1: Linear motion
    print("[TEST 1] Creating linear motion video...")
    path, gt = gen.create_linear_motion("test_linear.mp4", direction="right")
    assert Path(path).exists()
    assert len(gt) == 60  # 2 seconds at 30fps
    print("         PASSED!\n")

    # Test 2: Accelerating motion
    print("[TEST 2] Creating accelerating motion video...")
    path, gt = gen1.create_accelerating_motion("test_accel.mp4")
    assert Path(path).exists()
    # Should accelerate (later positions farther apart)
    dist_start = gt[1][0] - gt[0][0]
    dist_end = gt[-1][0] - gt[-2][0]
    assert dist_end > dist_start, "Should accelerate"
    print(f"         Start speed: {dist_start:.2f}, End speed: {dist_end:.2f}")
    print("         PASSED!\n")

    # Test 3: Multi-object
    print("[TEST 3] Creating multi-object video...")
    path, gts = gen.create_multi_object("test_multi.mp4", num_objects=3, crossing=True)
    assert Path(path).exists()
    assert len(gts) == 3
    print("         PASSED!\n")

    # Test 4: Route videos
    print("[TEST 4] Creating route videos...")
    for route in ["out", "curl", "slant"]:
        path, gt = gen1.create_route_video(f"test_{route}.mp4", route_type=route)
        assert Path(path).exists()
    print("         PASSED!\n")

    video_dir = Path("data/video_test")

    # saving to data folder just to check videos
    test_files = [
        "test_linear.mp4", 
        "test_accel.mp4", 
        "test_multi.mp4",
        "test_out.mp4", 
        "test_curl.mp4", 
        "test_slant.mp4"
    ]

    for filename in test_files:
        source_path = Path(filename)
        dest_path = video_dir / filename
        
        if source_path.exists():
            shutil.move(str(source_path), str(dest_path))
            print(f"Stored {filename} in data/video_test/")
        else:
            print(f"{filename} not found (maybe cleaned up?)")

    # COMMENT/UNCOMMENT TO DELETE TEST VIDEOS AFTER CHECKING
    # COMMENT/UNCOMMENT TO DELETE TEST VIDEOS AFTER CHECKING
    # COMMENT/UNCOMMENT TO DELETE TEST VIDEOS AFTER CHECKING
    # Cleanup
    # for filename in test_files:
    #     file_path = video_dir / filename
    #     if file_path.exists():
    #         file_path.unlink()

    print("="*60)
    print("All TestVideoGenerator tests passed!")
    print("="*60)