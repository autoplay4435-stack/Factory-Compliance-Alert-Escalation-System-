"""
Module 2 - Severity Routing Matrix.

Maps ComplianceEvent severity levels to the mandatory Module 3 routing action.

Routing rules:
  LOW / MEDIUM      -> persistent database log only.
  HIGH / CRITICAL   -> persistent database log and dashboard strobe alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import ComplianceEvent, SeverityLevel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    store_to_db: bool
    trigger_dashboard_strobe: bool
    escalation_action: str
    priority_label: str


_LOG_ONLY = "Logged to DB"
_ALERT_AND_LOG = "Real-time dashboard strobe triggered + DB log"


_ROUTING_TABLE: dict[SeverityLevel, RoutingDecision] = {
    SeverityLevel.LOW: RoutingDecision(
        store_to_db=True,
        trigger_dashboard_strobe=False,
        escalation_action=_LOG_ONLY,
        priority_label="Low Priority - Logged Only",
    ),
    SeverityLevel.MEDIUM: RoutingDecision(
        store_to_db=True,
        trigger_dashboard_strobe=False,
        escalation_action=_LOG_ONLY,
        priority_label="Medium Priority - Logged Only",
    ),
    SeverityLevel.HIGH: RoutingDecision(
        store_to_db=True,
        trigger_dashboard_strobe=True,
        escalation_action=_ALERT_AND_LOG,
        priority_label="High Priority - Dashboard Alert",
    ),
    SeverityLevel.CRITICAL: RoutingDecision(
        store_to_db=True,
        trigger_dashboard_strobe=True,
        escalation_action=_ALERT_AND_LOG,
        priority_label="Critical - Immediate Escalation",
    ),
}


def route_event(event: ComplianceEvent) -> RoutingDecision:
    decision = _ROUTING_TABLE.get(event.severity)
    if decision is None:
        logger.warning(
            "Unknown severity '%s' for event '%s', defaulting to LOW routing",
            event.severity,
            event.behavior_class,
        )
        decision = _ROUTING_TABLE[SeverityLevel.LOW]

    logger.debug(
        "Routing event '%s' (severity=%s): %s",
        event.behavior_class,
        event.severity.value,
        decision.priority_label,
    )
    return decision


def is_escalation_required(event: ComplianceEvent) -> bool:
    return event.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)