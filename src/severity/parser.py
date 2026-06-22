"""
Module 2 — Policy Parser + Severity Matrix.

Reads an unstructured regulatory policy document, makes a single OpenAI API call
with structured output to extract compliance rules, maps severity keywords to
severity levels, and writes the result to a JSON lookup table that Module 1
(Detection Engine) and Module 3 (Escalation Pipeline) consume.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.models import (
    DetectionType,
    PolicyRule,
    PolicyRuleSet,
    SeverityLevel,
)

load_dotenv()

logger = logging.getLogger(__name__)


class ExtractedRule(BaseModel):
    """Schema the LLM fills for each policy rule it finds."""

    behavior_class: str = Field(
        description="Short name of the compliance behavior class, e.g., 'PPE Compliance'"
    )
    observable_indicator: str = Field(
        description="What should be visually observed to detect a violation"
    )
    detection_type: str = Field(
        description=(
            "One of: hsv_color, contour_count, region_state, zone_boundary. "
            "Choose the technique that matches the observable indicator."
        )
    )
    detection_params: dict = Field(
        default_factory=dict,
        description=(
            "Detection parameters as key-value pairs. "
            "For hsv_color: lower_hsv=[H,S,V], upper_hsv=[H,S,V], min_pixel_ratio (float). "
            "For contour_count: expected_count (int), tolerance (int), roi_coords=[x,y,w,h]. "
            "For region_state: region_coords=[x,y,w,h], expected_state (str), tolerance (int). "
            "For zone_boundary: zone_polygon=[[x1,y1],[x2,y2],...], violation_direction (str)."
        ),
    )
    policy_section_ref: str = Field(
        description="The policy section reference, e.g., '§2' or '§4.2'"
    )
    severity_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Exact severity-related phrases found in the policy for this rule, "
            "e.g., ['CRITICAL SAFETY NOTICE', 'IMMEDIATE safety violation']"
        ),
    )


class ExtractedRuleList(BaseModel):
    """Wrapper for the list of rules the LLM extracts."""

    rules: list[ExtractedRule]


SEVERITY_KEYWORD_MAP: dict[str, SeverityLevel] = {
    "CRITICAL SAFETY NOTICE": SeverityLevel.CRIT,
    "CRITICAL": SeverityLevel.CRIT,
    "IMMEDIATE": SeverityLevel.CRIT,
    "imminent danger": SeverityLevel.CRIT,
    "WARNING": SeverityLevel.HIGH,
    "escalated": SeverityLevel.HIGH,
    "higher urgency": SeverityLevel.HIGH,
    "reported immediately": SeverityLevel.HIGH,
    "repeated": SeverityLevel.HIGH,
    "systemic": SeverityLevel.HIGH,
    "confirmed": SeverityLevel.MED,
    "anomaly": SeverityLevel.MED,
    "flagged": SeverityLevel.MED,
}

# Priority order: CRIT > HIGH > MED > LOW
_SEVERITY_PRIORITY = {
    SeverityLevel.LOW: 0,
    SeverityLevel.MED: 1,
    SeverityLevel.HIGH: 2,
    SeverityLevel.CRIT: 3,
}


def classify_severity(keywords: list[str]) -> SeverityLevel:
    best_severity = SeverityLevel.LOW
    best_priority = _SEVERITY_PRIORITY[SeverityLevel.LOW]

    for keyword in keywords:
        keyword_upper = keyword.upper()
        for pattern, severity in SEVERITY_KEYWORD_MAP.items():
            if pattern.upper() in keyword_upper:
                priority = _SEVERITY_PRIORITY[severity]
                if priority > best_priority:
                    best_severity = severity
                    best_priority = priority

    return best_severity


SYSTEM_PROMPT = """You are a compliance policy analyst. Your job is to read a factory safety policy document and extract every distinct compliance rule as structured data.

For each rule you find, extract:
1. behavior_class: A short, descriptive name (e.g., "PPE Compliance", "Material Stacking Limit")
2. observable_indicator: What a camera/vision system should look for to detect violations
3. detection_type: The computer vision technique to use:
   - "hsv_color" for color-based detection (e.g., vest color)
   - "contour_count" for counting objects (e.g., stacked blocks)
   - "region_state" for checking a fixed region's color/state (e.g., panel indicator)
   - "zone_boundary" for detecting if a person is inside/outside a zone
4. detection_params: Technical parameters for the detector:
   - For hsv_color: provide lower_hsv and upper_hsv as [H, S, V] arrays, and min_pixel_ratio as a float (0.0-1.0)
   - For contour_count: provide expected_count (int), tolerance (int), and roi_coords as [x, y, w, h]
   - For region_state: provide region_coords as [x, y, w, h], expected_state (str like "green"), and tolerance (int)
   - For zone_boundary: provide zone_polygon as [[x1,y1], [x2,y2], ...] and violation_direction ("inside" or "outside")
5. policy_section_ref: The exact section reference from the document (e.g., "§2", "§4.2")
6. severity_keywords: Extract the EXACT severity-related language from the policy text for this rule (e.g., "CRITICAL SAFETY NOTICE", "WARNING", "IMMEDIATE safety violation", "imminent danger")

Important:
- Extract ALL distinct rules, not just the first one.
- Use the observable indicator descriptions from the policy to set detection_params.
- For HSV ranges mentioned in the policy, convert them directly to the detection_params.
- For default parameters not specified in the policy, use reasonable defaults.
- Preserve the exact severity language from the document - do not paraphrase."""


class PolicyParser:

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the parser.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: OpenAI model to use for parsing.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model

        if not self._api_key:
            logger.warning(
                "No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key."
            )

    def parse(self, policy_text: str, source_name: str = "") -> PolicyRuleSet:
        """
        Parse a raw policy document into a structured PolicyRuleSet.

        Args:
            policy_text: The full text of the policy document.
            source_name: Optional name/path of the source document for provenance.

        Returns:
            A PolicyRuleSet containing all extracted and severity-classified rules.

        Raises:
            RuntimeError: If the LLM call fails or returns unparseable output.
        """
        extracted = self._call_llm(policy_text)
        rules = self._build_rules(extracted)

        return PolicyRuleSet(
            rules=rules,
            source_document=source_name,
        )

    def _call_llm(self, policy_text: str) -> ExtractedRuleList:
        """
        Make a single structured-output LLM call to extract rules.

        Args:
            policy_text: The raw policy document text.

        Returns:
            An ExtractedRuleList parsed from the LLM response.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            completion = client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract all compliance rules from the following "
                            "factory policy document:\n\n"
                            f"{policy_text}"
                        ),
                    },
                ],
                response_format=ExtractedRuleList,
            )

            parsed = completion.choices[0].message.parsed
            if parsed is None:
                # Model refused or failed to parse
                refusal = getattr(completion.choices[0].message, "refusal", None)
                raise RuntimeError(
                    f"LLM refused to parse the policy document: {refusal}"
                )

            logger.info("Successfully extracted %d rules from policy", len(parsed.rules))
            return parsed

        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"LLM policy parsing failed: {e}") from e

    def _build_rules(self, extracted: ExtractedRuleList) -> list[PolicyRule]:
        rules: list[PolicyRule] = []

        for raw in extracted.rules:
            # Validate detection_type
            try:
                detection_type = DetectionType(raw.detection_type)
            except ValueError:
                logger.warning(
                    "Unknown detection_type '%s' for rule '%s', defaulting to region_state",
                    raw.detection_type,
                    raw.behavior_class,
                )
                detection_type = DetectionType.REGION_STATE

            # Classify severity from keywords
            severity = classify_severity(raw.severity_keywords)

            # Apply default detection params if missing
            params = _apply_default_params(detection_type, raw.detection_params)

            rule = PolicyRule(
                behavior_class=raw.behavior_class,
                observable_indicator=raw.observable_indicator,
                detection_type=detection_type,
                detection_params=params,
                policy_section_ref=raw.policy_section_ref,
                severity_keywords=raw.severity_keywords,
                assigned_severity=severity,
            )
            rules.append(rule)

        return rules

    def parse_file(
        self, policy_path: str, output_path: Optional[str] = None
    ) -> PolicyRuleSet:
        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        policy_text = path.read_text(encoding="utf-8")
        rule_set = self.parse(policy_text, source_name=path.name)

        if output_path:
            rule_set.to_json_file(output_path)
            logger.info("Wrote parsed rules to %s", output_path)

        return rule_set


def _apply_default_params(
    detection_type: DetectionType, params: dict
) -> dict:
    defaults: dict[DetectionType, dict] = {
        DetectionType.HSV_COLOR: {
            # Green safety vest HSV range (Kafaoglu policy §4)
            "lower_hsv": [35, 80, 80],
            "upper_hsv": [85, 255, 255],
            "min_pixel_ratio": 0.15,
        },
        DetectionType.CONTOUR_COUNT: {
            # Safe carrying ≤ 2 blocks (Kafaoglu policy §6)
            "expected_count": 2,
            "tolerance": 0,
            "roi_coords": [0, 0, 640, 480],
        },
        DetectionType.REGION_STATE: {
            # Panel cover: expected state is "closed" (Kafaoglu policy §5)
            "region_coords": [0, 0, 100, 100],
            "expected_state": "closed",
            "tolerance": 30,
        },
        DetectionType.ZONE_BOUNDARY: {
            # Green walkway boundary (Kafaoglu policy §3)
            "zone_polygon": [[100, 400], [540, 400], [540, 480], [100, 480]],
            "violation_direction": "outside",
        },
    }

    type_defaults = defaults.get(detection_type, {})
    merged = {**type_defaults, **params}
    return merged


def create_fallback_rules() -> PolicyRuleSet:
    
    rules = [
        # Domain 0 — Pedestrian Movement (§3)
        PolicyRule(
            behavior_class="Safe Walkway Violation",
            observable_indicator="Personnel foot position outside green-marked walkway boundaries",
            detection_type=DetectionType.ZONE_BOUNDARY,
            detection_params={
                "zone_polygon": [[100, 400], [540, 400], [540, 470], [100, 470]],
                "violation_direction": "outside",
            },
            policy_section_ref="§3",
            severity_keywords=["WARNING", "highest-frequency unsafe behavior"],
            assigned_severity=SeverityLevel.HIGH,
        ),
        # Domain 1 — Equipment Intervention (§4)
        PolicyRule(
            behavior_class="Unauthorized Intervention",
            observable_indicator="Person interacting with equipment without wearing green safety vest",
            detection_type=DetectionType.HSV_COLOR,
            detection_params={
                # Green safety vest HSV range
                "lower_hsv": [35, 80, 80],
                "upper_hsv": [85, 255, 255],
                "min_pixel_ratio": 0.15,
            },
            policy_section_ref="§4",
            severity_keywords=["CRITICAL SAFETY NOTICE"],
            assigned_severity=SeverityLevel.CRIT,
        ),
        # Domain 2 — Electrical Safety (§5)
        PolicyRule(
            behavior_class="Opened Panel Cover",
            observable_indicator="Electrical panel cover in open position during production",
            detection_type=DetectionType.REGION_STATE,
            detection_params={
                "region_coords": [500, 50, 100, 100],
                "expected_state": "closed",
                "tolerance": 30,
            },
            policy_section_ref="§5",
            severity_keywords=["WARNING"],
            assigned_severity=SeverityLevel.HIGH,
        ),
        # Domain 3 — Forklift Load (§6)
        PolicyRule(
            behavior_class="Carrying Overload with Forklift",
            observable_indicator="Forklift carrying 3 or more standardized blocks",
            detection_type=DetectionType.CONTOUR_COUNT,
            detection_params={
                # Safe carrying ≤ 2, overload ≥ 3
                "expected_count": 2,
                "tolerance": 0,
                "roi_coords": [50, 200, 300, 250],
            },
            policy_section_ref="§6",
            severity_keywords=["CRITICAL SAFETY NOTICE"],
            assigned_severity=SeverityLevel.CRIT,
        ),
    ]

    return PolicyRuleSet(
        rules=rules,
        source_document="compliance_policy.pdf (fallback)",
    )
