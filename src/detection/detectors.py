"""
Module 1 — Detector Functions.

Four detector functions, each corresponding to one detection_type in the
PolicyRule schema. Each takes a video frame + a PolicyRule + optional landmarks
and returns an Optional[ComplianceEvent] if a violation is detected.

Detection techniques:
  1. HSV Color Thresholding  — vest/PPE detection
  2. Contour Counting        — material block stacking
  3. Region State Check      — control panel indicator
  4. Zone Boundary Test      — restricted area intrusion
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.models import BoundingBox, ComplianceEvent, PolicyRule, utc_now_iso

logger = logging.getLogger(__name__)


def _zone_from_rule(rule: PolicyRule) -> str:
    """Infer a stable facility zone label from rule metadata."""
    zone = rule.detection_params.get("zone") or rule.detection_params.get("zone_id")
    return str(zone or "Zone-1")


def detect_vest_color(
    frame: np.ndarray,
    rule: PolicyRule,
    pose_landmarks,
    frame_number: int,
) -> Optional[ComplianceEvent]:

    if pose_landmarks is None:
        return None

    params = rule.detection_params
    h, w, _ = frame.shape

    # Extract torso ROI from pose landmarks
    try:
        landmarks = pose_landmarks.landmark

        # Shoulder and hip landmarks define the torso
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Convert normalized coordinates to pixel coordinates
        x_min = int(min(left_shoulder.x, right_shoulder.x) * w)
        x_max = int(max(left_shoulder.x, right_shoulder.x) * w)
        y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
        y_max = int(max(left_hip.y, right_hip.y) * h)

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        # Ensure valid ROI
        if x_max <= x_min or y_max <= y_min:
            return None

        torso_roi = frame[y_min:y_max, x_min:x_max]

    except (IndexError, AttributeError):
        logger.debug("Could not extract torso ROI from pose landmarks")
        return None

    if torso_roi.size == 0:
        return None

    # Convert to HSV and apply color thresholding
    hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array(params.get("lower_hsv", [20, 100, 100]), dtype=np.uint8)
    upper_hsv = np.array(params.get("upper_hsv", [40, 255, 255]), dtype=np.uint8)
    min_pixel_ratio = params.get("min_pixel_ratio", 0.15)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Calculate the ratio of vest-colored pixels
    vest_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    pixel_ratio = vest_pixels / total_pixels if total_pixels > 0 else 0

    if pixel_ratio >= min_pixel_ratio:
        # Vest detected — compliant, no event
        return None

    # Vest NOT detected — violation
    confidence = max(0.0, min(1.0, 1.0 - pixel_ratio / min_pixel_ratio))

    return ComplianceEvent(
        timestamp=utc_now_iso(),
        zone=_zone_from_rule(rule),
        behavior_class=rule.behavior_class,
        policy_section_ref=rule.policy_section_ref,
        severity=rule.assigned_severity,
        confidence=round(confidence, 3),
        frame_number=frame_number,
        bounding_box=BoundingBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min),
        details=f"High-visibility vest not detected on torso (pixel ratio: {pixel_ratio:.3f}, required: {min_pixel_ratio})",
    )


def detect_block_count(
    frame: np.ndarray,
    rule: PolicyRule,
    frame_number: int,
) -> Optional[ComplianceEvent]:
    
    params = rule.detection_params
    h, w = frame.shape[:2]

    # Extract ROI
    roi_coords = params.get("roi_coords", [0, 0, w, h])
    rx, ry, rw, rh = roi_coords
    rx = max(0, min(rx, w))
    ry = max(0, min(ry, h))
    rw = min(rw, w - rx)
    rh = min(rh, h - ry)

    roi = frame[ry : ry + rh, rx : rx + rw]

    if roi.size == 0:
        return None

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold for varied lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small contours (noise)
    min_area = (rw * rh) * 0.01  # At least 1% of ROI
    significant_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    block_count = len(significant_contours)

    expected_count = params.get("expected_count", 5)
    tolerance = params.get("tolerance", 0)

    if block_count <= expected_count + tolerance:
        # Within limits — compliant
        return None

    # Overstacking detected
    excess = block_count - expected_count
    confidence = min(1.0, excess / (expected_count or 1))

    return ComplianceEvent(
        timestamp=utc_now_iso(),
        zone=_zone_from_rule(rule),
        behavior_class=rule.behavior_class,
        policy_section_ref=rule.policy_section_ref,
        severity=rule.assigned_severity,
        confidence=round(confidence, 3),
        frame_number=frame_number,
        bounding_box=BoundingBox(x=rx, y=ry, w=rw, h=rh),
        details=f"Block count {block_count} exceeds maximum of {expected_count} (excess: {excess})",
    )


def detect_panel_state(
    frame: np.ndarray,
    rule: PolicyRule,
    frame_number: int,
) -> Optional[ComplianceEvent]:
    """
    Detect control panel status by checking the dominant color in a fixed region.

    The expected state is typically "green" — if the region is red, dark, or
    not matching the expected color, a violation is reported.

    Args:
        frame: BGR video frame.
        rule: PolicyRule with detection_type=REGION_STATE.
        frame_number: Current frame index.

    Returns:
        ComplianceEvent if panel is in abnormal state, None if normal.
    """
    params = rule.detection_params
    h, w = frame.shape[:2]

    # Extract the panel indicator region
    region_coords = params.get("region_coords", [0, 0, 100, 100])
    rx, ry, rw, rh = region_coords
    rx = max(0, min(rx, w))
    ry = max(0, min(ry, h))
    rw = min(rw, w - rx)
    rh = min(rh, h - ry)

    region = frame[ry : ry + rh, rx : rx + rw]

    if region.size == 0:
        return None

    expected_state = params.get("expected_state", "green").lower()
    tolerance = params.get("tolerance", 30)

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]  # (H, S, V)

    mean_h, mean_s, mean_v = mean_hsv

    # Define expected color ranges
    state_ranges = {
        "green": {"h_range": (35, 85), "min_s": 50, "min_v": 50},
        "red": {"h_range": (0, 10), "min_s": 50, "min_v": 50},
        "blue": {"h_range": (100, 130), "min_s": 50, "min_v": 50},
    }

    expected = state_ranges.get(expected_state, state_ranges["green"])
    h_low, h_high = expected["h_range"]

    # Check if the region matches the expected color
    is_expected_color = (
        h_low - tolerance <= mean_h <= h_high + tolerance
        and mean_s >= expected["min_s"]
        and mean_v >= expected["min_v"]
    )

    if is_expected_color:
        # Panel is in expected state — compliant
        return None

    # Panel is in abnormal state
    # Determine what state it actually is
    actual_state = "unknown"
    if mean_v < 50:
        actual_state = "off/dark"
    elif 0 <= mean_h <= 10 or 160 <= mean_h <= 180:
        actual_state = "red"
    elif 35 <= mean_h <= 85:
        actual_state = "green"
    elif 100 <= mean_h <= 130:
        actual_state = "blue"
    elif 15 <= mean_h <= 35:
        actual_state = "yellow"

    confidence = 0.8 if mean_s > 80 else 0.6

    return ComplianceEvent(
        timestamp=utc_now_iso(),
        zone=_zone_from_rule(rule),
        behavior_class=rule.behavior_class,
        policy_section_ref=rule.policy_section_ref,
        severity=rule.assigned_severity,
        confidence=round(confidence, 3),
        frame_number=frame_number,
        bounding_box=BoundingBox(x=rx, y=ry, w=rw, h=rh),
        details=f"Control panel in '{actual_state}' state, expected '{expected_state}' (mean HSV: H={mean_h:.0f}, S={mean_s:.0f}, V={mean_v:.0f})",
    )


def detect_walkway_boundary(
    frame: np.ndarray,
    rule: PolicyRule,
    pose_landmarks,
    frame_number: int,
) -> Optional[ComplianceEvent]:
    
    if pose_landmarks is None:
        return None

    params = rule.detection_params
    h, w = frame.shape[:2]

    # Get the zone polygon
    zone_polygon = params.get("zone_polygon", [])
    if len(zone_polygon) < 3:
        logger.debug("Zone polygon has fewer than 3 points, skipping")
        return None

    polygon = np.array(zone_polygon, dtype=np.int32)
    violation_direction = params.get("violation_direction", "inside")

    try:
        landmarks = pose_landmarks.landmark

        # Check foot/ankle landmarks (31 = left foot, 32 = right foot)
        foot_landmarks = []
        for idx in [31, 32, 29, 30]:  # left foot, right foot, left heel, right heel
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.3:  # Only consider visible landmarks
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    foot_landmarks.append((px, py))

    except (IndexError, AttributeError):
        return None

    if not foot_landmarks:
        return None

    # Test each foot landmark against the polygon
    violations = []
    for px, py in foot_landmarks:
        result = cv2.pointPolygonTest(polygon, (float(px), float(py)), False)
        # result > 0: inside, result == 0: on edge, result < 0: outside

        is_inside = result >= 0
        is_violation = (
            (violation_direction == "inside" and is_inside)
            or (violation_direction == "outside" and not is_inside)
        )

        if is_violation:
            violations.append((px, py))

    if not violations:
        return None

    # Zone violation detected
    vx, vy = violations[0]
    poly_rect = cv2.boundingRect(polygon)

    return ComplianceEvent(
        timestamp=utc_now_iso(),
        zone=_zone_from_rule(rule),
        behavior_class=rule.behavior_class,
        policy_section_ref=rule.policy_section_ref,
        severity=rule.assigned_severity,
        confidence=0.95,  # Zone violations are high-confidence geometric tests
        frame_number=frame_number,
        bounding_box=BoundingBox(
            x=poly_rect[0], y=poly_rect[1], w=poly_rect[2], h=poly_rect[3]
        ),
        details=f"Personnel detected inside restricted zone at ({vx}, {vy}). {len(violations)} foot landmark(s) in violation area.",
    )


# Detector dispatch map: detection_type → detector function
DETECTOR_MAP = {
    "hsv_color": detect_vest_color,
    "contour_count": detect_block_count,
    "region_state": detect_panel_state,
    "zone_boundary": detect_walkway_boundary,
}
