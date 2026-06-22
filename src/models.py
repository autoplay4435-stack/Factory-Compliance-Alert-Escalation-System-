"""
Shared data models for the Factory Compliance and Alert Escalation System.

All modules import from this single source of truth for data structures.
Uses Pydantic for validation and serialization, with immutable (frozen) models
to enforce the immutability principle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return a UTC ISO 8601 timestamp with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def format_clip_timestamp(seconds: float) -> str:
    """Format seconds from the beginning of a clip as HH:MM:SS.mmm."""
    safe_seconds = max(0.0, float(seconds))
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    whole_seconds = int(safe_seconds % 60)
    milliseconds = int(round((safe_seconds - int(safe_seconds)) * 1000))

    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    if whole_seconds == 60:
        minutes += 1
        whole_seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def normalize_severity_value(value: str | SeverityLevel) -> str:
    """Normalize legacy and canonical severity labels for reports and filters."""
    if isinstance(value, SeverityLevel):
        return value.value

    normalized = str(value).upper()
    aliases = {
        "MED": "MEDIUM",
        "CRIT": "CRITICAL",
    }
    return aliases.get(normalized, normalized)


class SeverityLevel(str, Enum):
    """Severity classification for compliance events.

    Four-tier risk hierarchy derived from the Kafaoglu KMP-OHS-POL-001
    compliance policy document:

    LOW  — Low Risk: Condition observed but no immediate personnel
           proximity or imminent hazard. Typically a state-based finding
           (e.g., equipment condition) with no concurrent personnel exposure.
    MED  — Medium Risk: Behavioral deviation observed. Personnel present
           but not in immediate danger. Policy breach confirmed but hazard
           not yet acute.
    HIGH — High Risk: Active unsafe behavior with concurrent personnel
           exposure or operational risk. Hazard is present and could
           result in injury.
    CRIT — Critical Risk: Immediate danger condition. Most severe policy
           breach; either high-frequency recurrence, direct injury risk,
           or the behavior category is explicitly flagged as the highest-
           consequence hazard in the policy.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    MED = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    CRIT = "CRITICAL"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = normalize_severity_value(value)
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class DetectionType(str, Enum):
    """
    The type of computer vision technique used to detect a violation.
    Each maps to a specific detector function in the detection engine.
    """

    HSV_COLOR = "hsv_color"
    CONTOUR_COUNT = "contour_count"
    REGION_STATE = "region_state"
    ZONE_BOUNDARY = "zone_boundary"


class PolicyRule(BaseModel):
    """
    One parsed compliance rule extracted from the regulatory policy document.

    This is the central data structure that bridges Module 2 (Policy Parser)
    to Module 1 (Detection Engine) and Module 3 (Escalation Pipeline).
    Each rule describes what to detect, how to detect it, what policy section
    it traces back to, and how severe a violation is.
    """

    model_config = {"frozen": True}

    behavior_class: str = Field(
        description="Name of the compliance behavior class, e.g., 'PPE Compliance'"
    )
    observable_indicator: str = Field(
        description="What the detector should look for, e.g., 'high-visibility vest on torso'"
    )
    detection_type: DetectionType = Field(
        description="Which detection technique to use"
    )
    detection_params: dict = Field(
        default_factory=dict,
        description=(
            "Parameters for the detector function. Keys vary by detection_type: "
            "hsv_color: lower_hsv, upper_hsv, min_pixel_ratio; "
            "contour_count: expected_count, tolerance, roi_coords; "
            "region_state: region_coords, expected_state, tolerance; "
            "zone_boundary: zone_polygon, violation_direction"
        ),
    )
    policy_section_ref: str = Field(
        description="Policy section reference, e.g., '§4.2'"
    )
    severity_keywords: list[str] = Field(
        default_factory=list,
        description="Raw severity language from the policy, e.g., ['CRITICAL SAFETY NOTICE']",
    )
    assigned_severity: SeverityLevel = Field(
        description="Computed severity level for this rule"
    )


class PolicyRuleSet(BaseModel):
    """Container for a complete set of parsed policy rules."""

    model_config = {"frozen": True}

    rules: list[PolicyRule] = Field(default_factory=list)
    source_document: str = Field(
        default="", description="Name/path of the source policy document"
    )
    parsed_at: str = Field(
        default_factory=utc_now_iso,
        description="ISO 8601 timestamp of when the parsing was done",
    )

    def to_json_file(self, filepath: str) -> None:
        """Serialize the rule set to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, filepath: str) -> PolicyRuleSet:
        """Deserialize a rule set from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected region."""

    model_config = {"frozen": True}

    x: int = Field(ge=0, description="Top-left x coordinate")
    y: int = Field(ge=0, description="Top-left y coordinate")
    w: int = Field(ge=0, description="Width in pixels")
    h: int = Field(ge=0, description="Height in pixels")


class ComplianceEvent(BaseModel):
    """
    One detected compliance violation event.

    This is the primary data object flowing through the escalation pipeline.
    Created by Module 1 detectors, routed by Module 3, stored by Module 4.
    """

    model_config = {"frozen": True}

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this violation event",
    )
    clip_id: str = Field(
        default="unknown-clip", description="Identifier for the source video clip"
    )
    zone: str = Field(
        default="Zone-1",
        description="Facility zone where the event occurred",
    )
    timestamp: str = Field(
        default_factory=utc_now_iso,
        description="ISO 8601 timestamp of when the event was detected",
    )
    clip_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Timestamp of the event relative to the start of the source clip",
    )
    clip_timestamp: str = Field(
        default="00:00:00.000",
        description="Formatted clip-relative timestamp as HH:MM:SS.mmm",
    )
    behavior_class: str = Field(
        description="Which policy rule was violated"
    )
    policy_section_ref: str = Field(
        description="Traceable reference to the policy section"
    )
    severity: SeverityLevel = Field(
        description="Severity level of the violation"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Detection confidence score"
    )
    frame_number: int = Field(
        ge=0, description="Video frame index where violation was detected"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None, description="Region of the frame where violation was detected"
    )
    details: str = Field(
        default="", description="Human-readable description of the violation"
    )

    @property
    def policy_rule_ref(self) -> str:
        """Report-field alias for the policy reference."""
        return self.policy_section_ref

    @property
    def event_description(self) -> str:
        """Report-field alias for the human-readable event description."""
        return self.details

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        data = self.model_dump()
        data["severity"] = self.severity.value
        if self.bounding_box is not None:
            data["bounding_box"] = self.bounding_box.model_dump()
        else:
            data["bounding_box"] = None
        return data

    def to_report_dict(self, escalation_action: str) -> dict:
        """Convert to the immutable compliance report schema required by Module 4."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "clip_id": self.clip_id,
            "zone": self.zone,
            "behavior_class": self.behavior_class,
            "policy_rule_ref": self.policy_rule_ref,
            "event_description": self.event_description,
            "severity": self.severity.value,
            "escalation_action": escalation_action,
        }
