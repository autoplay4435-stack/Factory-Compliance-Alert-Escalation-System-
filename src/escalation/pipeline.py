"""
Module 3 - Escalation Pipeline.

Routes each detected violation according to its severity tier with a direct
database-backed workflow.

Mandatory routing rules:
  LOW / MEDIUM      -> write the immutable database report only.
  HIGH / CRITICAL   -> write the immutable database report and mark the
                       dashboard strobe alert as active for that event.
"""

from __future__ import annotations

import logging
import threading

from src.models import ComplianceEvent
from src.severity.matrix import RoutingDecision, route_event

logger = logging.getLogger(__name__)


class EscalationPipeline:

    def __init__(self, database):
        self._database = database
        self._lock = threading.Lock()
        self._event_count = 0
        self._alert_count = 0

    def route(self, event: ComplianceEvent) -> RoutingDecision:

        decision = route_event(event)

        if decision.store_to_db:
            self._database.insert_event(
                event,
                escalated=decision.trigger_dashboard_strobe,
                escalation_action=decision.escalation_action,
            )

        with self._lock:
            self._event_count += 1
            if decision.trigger_dashboard_strobe:
                self._alert_count += 1

        if decision.trigger_dashboard_strobe:
            logger.warning(
                "Dashboard strobe alert active: [%s] %s",
                event.severity.value,
                event.behavior_class,
            )
        else:
            logger.info(
                "Violation logged: [%s] %s",
                event.severity.value,
                event.behavior_class,
            )

        return decision

    def route_many(self, events: list[ComplianceEvent]) -> list[RoutingDecision]:
        return [self.route(event) for event in events]

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._event_count

    @property
    def alert_count(self) -> int:
        with self._lock:
            return self._alert_count