"""
Module 1 — Detection Engine (Main Loop).

Orchestrates the video processing pipeline:
  1. Loads policy rules from JSON
  2. Initializes MediaPipe Pose
  3. Processes frames through the appropriate detectors
  4. Routes detected events through the escalation pipeline
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.detection.detectors import DETECTOR_MAP
from src.detection.video_source import VideoSource
from src.escalation.pipeline import EscalationPipeline
from src.models import DetectionType, PolicyRule, PolicyRuleSet, format_clip_timestamp

logger = logging.getLogger(__name__)


class DetectionEngine:

    def __init__(
        self,
        rules: PolicyRuleSet,
        escalation_pipeline: EscalationPipeline,
        process_every_n_frames: int = 3,
        display: bool = False,
    ):

        self._rules = rules
        self._escalation_pipeline = escalation_pipeline
        self._process_every_n = max(1, process_every_n_frames)
        self._display = display
        self._running = False
        self._pose = None
        self._clip_id = ""

        # Separate rules by detection type for efficient dispatch
        self._pose_rules: list[PolicyRule] = []
        self._frame_rules: list[PolicyRule] = []

        for rule in rules.rules:
            if rule.detection_type in (
                DetectionType.HSV_COLOR,
                DetectionType.ZONE_BOUNDARY,
            ):
                self._pose_rules.append(rule)
            else:
                self._frame_rules.append(rule)

        logger.info(
            "Detection engine initialized: %d pose rules, %d frame rules, "
            "processing every %d frames",
            len(self._pose_rules),
            len(self._frame_rules),
            self._process_every_n,
        )

    def _init_mediapipe(self):
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self._mp_pose = mp.solutions.pose
            else:
                import mediapipe.python.solutions.pose as pose

                self._mp_pose = pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose initialized")
        except ImportError:
            logger.warning(
                "MediaPipe not available — pose-based detectors will be skipped"
            )
            self._pose = None

    def run(self, source: str | int = 0) -> None:
        self._init_mediapipe()
        self._running = True
        events_detected = 0

        # Derive clip identifier from source path/index
        from pathlib import Path
        self._clip_id = Path(str(source)).stem if isinstance(source, str) else f"webcam_{source}"

        with VideoSource(source) as video:
            if not video._cap or not video._cap.isOpened():
                logger.error("Cannot open video source: %s", source)
                return

            logger.info("Starting detection on source: %s (clip_id=%s)", source, self._clip_id)
            fps_start = time.time()

            while self._running:
                ret, frame = video.read()
                if not ret:
                    logger.info("End of video stream")
                    break

                frame_num = video.frame_number

                # Throttle: only process every Nth frame
                if frame_num % self._process_every_n != 0:
                    continue

                # Process the frame
                new_events = self._process_frame(
                    frame,
                    frame_num,
                    clip_time_seconds=video.clip_time_seconds,
                )
                events_detected += len(new_events)

                # Route events
                for event in new_events:
                    self._escalation_pipeline.route(event)

                # Display annotated frame if requested
                if self._display:
                    self._draw_annotations(frame, new_events)
                    cv2.imshow("Factory Compliance Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                        logger.info("User pressed ESC, stopping")
                        break

            elapsed = time.time() - fps_start
            logger.info(
                "Detection complete: %d frames processed, %d events detected in %.1fs",
                frame_num,
                events_detected,
                elapsed,
            )

        if self._display:
            cv2.destroyAllWindows()

        if self._pose is not None:
            self._pose.close()

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        clip_id: str = "",
        clip_time_seconds: float = 0.0,
    ) -> list:
       
        effective_clip_id = clip_id or self._clip_id
        clip_timestamp = format_clip_timestamp(clip_time_seconds)
        events = []
        pose_landmarks = None

        # Run MediaPipe Pose if we have pose-dependent rules
        if self._pose is not None and self._pose_rules:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb_frame)
            pose_landmarks = results.pose_landmarks if results else None

        # Run pose-dependent detectors (vest color, zone boundary)
        for rule in self._pose_rules:
            detector_key = rule.detection_type.value
            detector_fn = DETECTOR_MAP.get(detector_key)

            if detector_fn is None:
                continue

            try:
                event = detector_fn(
                    frame=frame,
                    rule=rule,
                    pose_landmarks=pose_landmarks,
                    frame_number=frame_number,
                )
                if event is not None:
                    event = event.model_copy(
                        update={
                            "clip_id": effective_clip_id,
                            "clip_time_seconds": round(clip_time_seconds, 3),
                            "clip_timestamp": clip_timestamp,
                        }
                    )
                    events.append(event)
            except Exception:
                logger.exception(
                    "Detector '%s' failed on frame %d",
                    detector_key,
                    frame_number,
                )

        # Run frame-only detectors (block count, panel state)
        for rule in self._frame_rules:
            detector_key = rule.detection_type.value
            detector_fn = DETECTOR_MAP.get(detector_key)

            if detector_fn is None:
                continue

            try:
                event = detector_fn(
                    frame=frame,
                    rule=rule,
                    frame_number=frame_number,
                )
                if event is not None:
                    event = event.model_copy(
                        update={
                            "clip_id": effective_clip_id,
                            "clip_time_seconds": round(clip_time_seconds, 3),
                            "clip_timestamp": clip_timestamp,
                        }
                    )
                    events.append(event)
            except Exception:
                logger.exception(
                    "Detector '%s' failed on frame %d",
                    detector_key,
                    frame_number,
                )

        return events

    def _draw_annotations(self, frame: np.ndarray, events: list) -> None:
        
        severity_colors = {
            "LOW": (0, 255, 0),      # Green
            "MED": (0, 255, 255),    # Yellow
            "HIGH": (0, 165, 255),   # Orange
            "CRIT": (0, 0, 255),     # Red
        }

        for event in events:
            color = severity_colors.get(event.severity.value, (255, 255, 255))

            if event.bounding_box is not None:
                bb = event.bounding_box
                cv2.rectangle(
                    frame,
                    (bb.x, bb.y),
                    (bb.x + bb.w, bb.y + bb.h),
                    color,
                    2,
                )

            # Label
            label = f"[{event.severity.value}] {event.behavior_class}"
            y_offset = event.bounding_box.y - 10 if event.bounding_box else 30
            x_offset = event.bounding_box.x if event.bounding_box else 10
            cv2.putText(
                frame,
                label,
                (x_offset, max(y_offset, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def stop(self) -> None:
        
        self._running = False
        logger.info("Detection engine stop requested")

    def process_single_frame(
        self, frame: np.ndarray, frame_number: int = 0
    ) -> list:
       
        if self._pose is None:
            self._init_mediapipe()
        return self._process_frame(frame, frame_number)