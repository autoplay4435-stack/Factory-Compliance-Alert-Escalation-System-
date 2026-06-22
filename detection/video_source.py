"""
Module 1 — Video Source Abstraction.

Thin wrapper around cv2.VideoCapture that supports webcam index,
video file path, and provides frame metadata.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:

    def __init__(self, source: str | int = 0):
        """
        Initialize the video source.

        Args:
            source: Either a file path (str) to a video file, or an integer
                    index for a webcam (0 = default camera).
        """
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_number = 0

    def open(self) -> bool:
    
        self._cap = cv2.VideoCapture(self._source)
        is_opened = self._cap.isOpened()

        if is_opened:
            logger.info(
                "Video source opened: %s (%.0f FPS, %dx%d)",
                self._source,
                self.fps,
                self.width,
                self.height,
            )
        else:
            logger.error("Failed to open video source: %s", self._source)

        return is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_number += 1

        return ret, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Video source released: %s", self._source)

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def clip_time_seconds(self) -> float:
        if self._cap is None:
            return 0.0

        position_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms and position_ms > 0:
            return position_ms / 1000.0

        fps = self.fps or 30.0
        return max(0.0, (self._frame_number - 1) / fps)

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
