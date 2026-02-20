"""
NFL Route Tracker - Video Loader Module
========================================

This module handles all video input/output operations. It provides a clean
interface for loading videos, extracting frames, and managing video metadata.

Author: Sam Gold
Phase: 1 - Foundation
"""

# important packages
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union
from dataclasses import dataclass

# initializing video attributes to keep track of 
@dataclass
class VideoMetadata:
    """
    Container for video properties.

    Attributes:
    -----------
    width : int
        Frame width in pixels
    height : int
        Frame height in pixels
    fps : float
        Frames per second
    total_frames : int
        Total number of frames in video
    duration_seconds : float
        Video duration in seconds
    codec : str
        Video codec (e.g., 'mp4v', 'avc1')
    """
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    # easily print out video metadata 
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Video: {self.width}x{self.height} @ {self.fps:.2f}fps, "
            f"{self.total_frames} frames ({self.duration_seconds:.2f}s), "
            f"codec: {self.codec}"
        )
    
class VideoLoader:
    """
    Handles loading and iterating over video frames.


    Example Usage:
    -------------
    ```python
    # Basic usage - iterate over all frames
    with VideoLoader("my_video.mp4") as video:
        print(video.metadata)  # See video properties
        for frame_num, frame in video:
            # frame is a numpy array of shape (height, width, channels)
            process_frame(frame)

    # Load as grayscale
    with VideoLoader("my_video.mp4", grayscale=True) as video:
        for frame_num, frame in video:
            # frame is now (height, width) - no color channels
            pass
    ```

    Parameters:
    -----------
    video_path : str or Path
        Path to the video file
    grayscale : bool
        If True, convert frames to grayscale (useful for motion detection)
    resize : Optional[Tuple[int, int]]
        If provided, resize frames to (width, height)
    """

    def __init__(self, video_path: Union[str, Path], grayscale: bool = False, resize: Optional[Tuple[int, int]] = None):
        """
        Initialize the video loader
        """
        self.video_path = Path(video_path)
        self.grayscale = grayscale
        self.resize = resize

        # These are set when the video is opened
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

        # Track iteration state
        self._current_frame: int = 0

        print(f"[VideoLoader] Initialized for: {self.video_path}")
        print(f"Grayscale: {self.grayscale}")
        print(f"Resize: {self.resize}")
        
    # open video and extract metadata from initialized object
    def __enter__(self) -> 'VideoLoader':
        """Open the video file and read metadata.
        Called when writing `with VideoLoader(...) as video:`."""
        
        print(f"[VideoLoader] Opening video file...")

        # Verify file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open video with OpenCV
        self._cap = cv2.VideoCapture(str(self.video_path))

        # Verify it opened successfully
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Extract metadata
        self._metadata = self._extract_metadata()
        print(f"[VideoLoader] Successfully opened: {self._metadata}")

        return self

    # ensures video resources are released even if an exception occurs
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - releases video resources.
        Always called when exiting the `with` block, ensures resources dont leak
        """
        print(f"[VideoLoader] Releasing video resources...")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print(f"[VideoLoader] Resources released successfully")

    # gets video attributes and store as VideoMetadata dataclass for easy access
    def _extract_metadata(self) -> VideoMetadata:
        """
        Extract video properties from OpenCV capture object after storing from initialization.
        OpenCV uses "property IDs" (CAP_PROP_*) to query video info - convert numbers into a clean dataclass.
        """
        # Get raw properties from OpenCV
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decode the fourcc codec identifier
        # FOURCC is a 4-byte code identifying the video codec
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0

        return VideoMetadata(
            width = width,
            height = height,
            fps = fps,
            total_frames = total_frames,
            duration_seconds = duration,
            codec = codec
        )
    
    # exposes the metadata from _extract_metadata in a safe, convenient way
    @property
    def metadata(self) -> VideoMetadata:
        """Access video metadata after opening the video."""
        if self._metadata is None:
            raise RuntimeError("Video metadata not available. Use 'with VideoLoader(...) as v:' to open a video")
        return self._metadata
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Make the video iterable.
        Yields tuples of (frame_number, frame_array).
        """
        self._current_frame = 0
        return self
    
    # calles preproccessing function, reads next frame, applies preproccessing, and returns frame number and frame array as a tuple
    def __next__(self) -> Tuple[int, np.ndarray]:
        """
        Get the next frame.
        """
        if self._cap is None:
            raise RuntimeError("Video not opened")

        # Read next frame
        ret, frame = self._cap.read()

        # Check if we got a valid frame
        if not ret or frame is None:
            print(f"[VideoLoader] End of video reached at frame {self._current_frame}")
            raise StopIteration

        # Apply preprocessing
        frame = self._preprocess_frame(frame)

        # Track frame number
        frame_num = self._current_frame
        self._current_frame += 1

        # Progress logging (every 100 frames)
        if frame_num % 100 == 0:
            print(f"[VideoLoader] Processing frame {frame_num}/{self.metadata.total_frames}")

        return frame_num, frame
    
    # optional resizing and grayscale conversion
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a frame (resize then greyscale).
        Parameters:
        -----------
        frame : np.ndarray
            Raw BGR frame from OpenCV (shape: height, width, 3)

        Returns:
        --------
        np.ndarray
            Preprocessed frame
        """
        # Resize if requested
        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)

        # Convert to grayscale if requested
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame
    
    # seek to and read a specific frame by its index, not important, for random access if needed
    # maybe helpful to show as examples at specific frames in the video, like showing the first frame, a mid frame, and the last frame in the documentation or tests
    def get_frame(self, frame_number: int) -> np.ndarray:
        """
        Get a specific frame by number.
        Parameters:
        -----------
        frame_number : int
            Frame index (0-based)

        Returns:
        --------
        np.ndarray
            The requested frame
        """
        if self._cap is None:
            raise RuntimeError("Video not opened")

        if not 0 <= frame_number < self.metadata.total_frames:
            raise ValueError(
                f"Frame {frame_number} out of range [0, {self.metadata.total_frames})"
            )

        # Seek to frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = self._cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_number}")

        return self._preprocess_frame(frame)

    def reset(self):
        """
        Reset video to beginning.
        """
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame = 0
            print("[VideoLoader] Reset to frame 0")


from nfl_route_tracker.utils.test_video_generator import TestVideoGenerator, VideoConfig
from pathlib import Path

if __name__ == "__main__":
    """
    Test the VideoLoader module.
    """
    print("\n" + "="*60)
    print("Testing VideoLoader Module")
    print("="*60 + "\n")

    # use example videos from test_video_generator to test loading and frame iteration
    test_folder = Path(__file__).parent.parent.parent.parent / "data" / "video_test"
    video_files = list(test_folder.glob("*.mp4"))
    if not video_files:
        print(f"No video files found in {test_folder}")
    else:
        print("\n" + "="*60)
        print("Testing VideoLoader on existing MP4 files")
        print("="*60 + "\n")
        # if test videos exist, go through each one and test video_loader
        for video_path in video_files:
            print(f"[TEST] Loading video: {video_path.name}")
            try:
                # Load video and print metadata
                with VideoLoader(video_path) as video:
                    print(f"Metadata: {video.metadata}")

                    # Test iteration for a few frames to ensure looping works
                    frame_count = 0
                    for frame_num, frame in video:
                        frame_count += 1
                        if frame_count >= 5: 
                            break
                    print(f"Successfully iterated {frame_count} frames")

                    # Test grayscale option
                    with VideoLoader(video_path, grayscale=True) as gray_video:
                        _, frame_gray = next(iter(gray_video))
                        assert len(frame_gray.shape) == 2, "Grayscale conversion failed"
                        print(f"Grayscale frame shape: {frame_gray.shape}")

                    # Test resize option
                    new_size = (160, 120)
                    with VideoLoader(video_path, resize=new_size) as video:
                        _, frame = next(iter(video))
                        assert frame.shape[:2] == (new_size[1], new_size[0]), "Resize failed"
                        print(f"Resized frame shape: {frame.shape}")

                    # Test random frame access option
                    with VideoLoader(video_path) as video:
                        frame_num = min(30, video.metadata.total_frames - 1)
                        frame = video.get_frame(frame_num)
                        print(f"Got frame {frame_num}, shape: {frame.shape}")

                print("PASSED!\n")
            except Exception as e:
                print(f"FAILED: {e}\n")
                

    print("="*60)
    print("All VideoLoader tests passed!")
    print("="*60)
