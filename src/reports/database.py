"""
Module 4 - Automated Report Generation.

SQLite persistence for immutable compliance records. Every detected violation
that reaches Module 3 is written here automatically with the required audit
fields and enough detector metadata to support dashboard review.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.models import ComplianceEvent, normalize_severity_value

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/compliance_events.db"

LOG_ONLY_ACTION = "Logged to DB"
ALERT_AND_LOG_ACTION = "Real-time dashboard strobe triggered + DB log"

REPORT_FIELDNAMES = [
    "event_id",
    "timestamp",
    "clip_id",
    "zone",
    "behavior_class",
    "policy_rule_ref",
    "event_description",
    "severity",
    "escalation_action",
]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS compliance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    zone TEXT NOT NULL,
    behavior_class TEXT NOT NULL,
    policy_rule_ref TEXT NOT NULL,
    policy_section_ref TEXT NOT NULL,
    event_description TEXT NOT NULL,
    severity TEXT NOT NULL,
    escalation_action TEXT NOT NULL,
    clip_time_seconds REAL NOT NULL DEFAULT 0.0,
    clip_timestamp TEXT NOT NULL DEFAULT '00:00:00.000',
    confidence REAL NOT NULL,
    frame_number INTEGER NOT NULL,
    bounding_box TEXT,
    details TEXT NOT NULL DEFAULT '',
    escalated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_INDEX_SQL = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_event_id ON compliance_events(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_severity ON compliance_events(severity)",
    "CREATE INDEX IF NOT EXISTS idx_behavior_class ON compliance_events(behavior_class)",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON compliance_events(timestamp)",
]

_MIGRATION_COLUMNS = {
    "event_id": "TEXT",
    "clip_id": "TEXT NOT NULL DEFAULT 'unknown-clip'",
    "zone": "TEXT NOT NULL DEFAULT 'Zone-1'",
    "policy_rule_ref": "TEXT NOT NULL DEFAULT ''",
    "policy_section_ref": "TEXT NOT NULL DEFAULT ''",
    "event_description": "TEXT NOT NULL DEFAULT ''",
    "details": "TEXT NOT NULL DEFAULT ''",
    "escalation_action": "TEXT NOT NULL DEFAULT ''",
    "clip_time_seconds": "REAL NOT NULL DEFAULT 0.0",
    "clip_timestamp": "TEXT NOT NULL DEFAULT '00:00:00.000'",
    "escalated": "INTEGER NOT NULL DEFAULT 0",
}

_INSERT_EVENT_SQL = """
INSERT INTO compliance_events
    (event_id, timestamp, clip_id, zone, behavior_class, policy_rule_ref,
     policy_section_ref, event_description, severity, escalation_action,
     clip_time_seconds, clip_timestamp, confidence, frame_number, bounding_box,
     details, escalated)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_EVENTS_SQL = "SELECT * FROM compliance_events"


class ComplianceDatabase:
    """
    SQLite-backed storage for compliance reports.

    Thread-safe writes are serialized with a lock. All filters use
    parameterized queries.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        """Create or migrate the compliance report table."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(_CREATE_TABLE_SQL)
                self._migrate_existing_table(cursor)
                for index_sql in _CREATE_INDEX_SQL:
                    cursor.execute(index_sql)
                conn.commit()
                logger.info("Database initialized at %s", self._db_path)
            finally:
                conn.close()

    def _migrate_existing_table(self, cursor: sqlite3.Cursor) -> None:
        """Add required report columns when opening an older database."""
        cursor.execute("PRAGMA table_info(compliance_events)")
        columns = {row[1] for row in cursor.fetchall()}

        for name, definition in _MIGRATION_COLUMNS.items():
            if name not in columns:
                cursor.execute(f"ALTER TABLE compliance_events ADD COLUMN {name} {definition}")

        cursor.execute("SELECT id FROM compliance_events WHERE event_id IS NULL OR event_id = ''")
        for (row_id,) in cursor.fetchall():
            cursor.execute(
                "UPDATE compliance_events SET event_id = ? WHERE id = ?",
                (str(uuid4()), row_id),
            )

        cursor.execute(
            """
            UPDATE compliance_events
            SET severity = CASE severity
                WHEN 'MED' THEN 'MEDIUM'
                WHEN 'CRIT' THEN 'CRITICAL'
                ELSE severity
            END
            """
        )
        cursor.execute(
            """
            UPDATE compliance_events
            SET policy_rule_ref = policy_section_ref
            WHERE policy_rule_ref IS NULL OR policy_rule_ref = ''
            """
        )
        cursor.execute(
            """
            UPDATE compliance_events
            SET event_description = details
            WHERE event_description IS NULL OR event_description = ''
            """
        )
        cursor.execute(
            """
            UPDATE compliance_events
            SET escalation_action = CASE
                WHEN escalated = 1 THEN ?
                ELSE ?
            END
            WHERE escalation_action IS NULL OR escalation_action = ''
            """,
            (ALERT_AND_LOG_ACTION, LOG_ONLY_ACTION),
        )

    def insert_event(
        self,
        event: ComplianceEvent,
        escalated: bool = False,
        escalation_action: Optional[str] = None,
    ) -> int:

        bbox_json = (
            json.dumps(event.bounding_box.model_dump())
            if event.bounding_box is not None
            else None
        )
        action = escalation_action or (
            ALERT_AND_LOG_ACTION if escalated else LOG_ONLY_ACTION
        )

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    _INSERT_EVENT_SQL,
                    (
                        event.event_id,
                        event.timestamp,
                        event.clip_id,
                        event.zone,
                        event.behavior_class,
                        event.policy_rule_ref,
                        event.policy_section_ref,
                        event.event_description,
                        normalize_severity_value(event.severity),
                        action,
                        event.clip_time_seconds,
                        event.clip_timestamp,
                        event.confidence,
                        event.frame_number,
                        bbox_json,
                        event.details,
                        1 if escalated else 0,
                    ),
                )
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug("Inserted report id=%d event_id=%s", row_id, event.event_id)
                return row_id
            finally:
                conn.close()

    def get_events(
        self,
        severity: Optional[str] = None,
        behavior_class: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """
        Query compliance events with optional filters.
        """
        conditions: list[str] = []
        params: list = []

        if severity:
            conditions.append("severity = ?")
            params.append(normalize_severity_value(severity))
        if behavior_class:
            conditions.append("behavior_class = ?")
            params.append(behavior_class)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        query = _SELECT_EVENTS_SQL
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_report_events(self, **filters) -> list[dict]:
        return [self.to_report_record(row) for row in self.get_events(**filters)]

    def get_recent_events(self, count: int = 50) -> list[dict]:
        return self.get_events(limit=count)

    def get_recent_alerts(self, count: int = 20) -> list[dict]:
        query = (
            _SELECT_EVENTS_SQL
            + " WHERE escalated = 1 OR severity IN (?, ?) "
            + "ORDER BY timestamp DESC LIMIT ?"
        )
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, ("HIGH", "CRITICAL", count))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_by_severity(self) -> dict[str, int]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT severity, COUNT(*) as count
                FROM compliance_events
                GROUP BY severity
                """
            )
            rows = cursor.fetchall()
            return {normalize_severity_value(row[0]): row[1] for row in rows}
        finally:
            conn.close()

    def count_by_behavior(self) -> dict[str, int]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT behavior_class, COUNT(*) as count
                FROM compliance_events
                GROUP BY behavior_class
                """
            )
            rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            conn.close()

    def total_count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM compliance_events")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def to_report_record(self, row: dict) -> dict:
        escalated = bool(row.get("escalated", 0))
        return {
            "event_id": row.get("event_id") or str(row.get("id", "")),
            "timestamp": row.get("timestamp", ""),
            "clip_id": row.get("clip_id") or "unknown-clip",
            "zone": row.get("zone") or "Zone-1",
            "behavior_class": row.get("behavior_class", ""),
            "policy_rule_ref": row.get("policy_rule_ref")
            or row.get("policy_section_ref", ""),
            "event_description": row.get("event_description")
            or row.get("details", ""),
            "severity": normalize_severity_value(row.get("severity", "LOW")),
            "escalation_action": row.get("escalation_action")
            or (ALERT_AND_LOG_ACTION if escalated else LOG_ONLY_ACTION),
        }

    def export_csv(self, filepath: str, **filters) -> int:
        events = self.get_report_events(limit=100000, **filters)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if not events:
                return 0
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(events)

        logger.info("Exported %d reports to %s", len(events), filepath)
        return len(events)

    def export_csv_string(self, **filters) -> str:
        events = self.get_report_events(limit=100000, **filters)
        if not events:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(events)
        return output.getvalue()

    def close(self) -> None:
        logger.info("Database handle closed for %s", self._db_path)