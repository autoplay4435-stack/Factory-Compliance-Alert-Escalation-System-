"""
Module 5 - Operations Dashboard.

Streamlit dashboard for monitoring compliance status, reviewing alert history,
and exporting immutable audit records from SQLite.
"""

from __future__ import annotations

import csv
import html
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.reports.database import ComplianceDatabase, REPORT_FIELDNAMES


st.set_page_config(
    page_title="Factory Compliance Monitor",
    page_icon="FC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .feed-panel {
        min-height: 260px;
        border: 1px solid #2f3847;
        border-radius: 8px;
        background: linear-gradient(135deg, #111827 0%, #202733 100%);
        color: #f8fafc;
        padding: 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .feed-grid {
        height: 150px;
        border: 1px solid rgba(255,255,255,.18);
        background-image:
            linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px);
        background-size: 24px 24px;
        border-radius: 6px;
        position: relative;
        overflow: hidden;
    }
    .feed-zone {
        position: absolute;
        left: 18%;
        bottom: 18%;
        width: 58%;
        height: 24%;
        border: 2px solid #22c55e;
        background: rgba(34,197,94,.12);
    }
    .feed-target {
        position: absolute;
        right: 18%;
        top: 22%;
        width: 64px;
        height: 54px;
        border: 2px solid currentColor;
        border-radius: 6px;
    }
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
    }
    .status-low { background: #16a34a; }
    .status-medium { background: #2563eb; }
    .status-high { background: #d97706; }
    .status-critical { background: #dc2626; }
    .status-clear { background: #475569; }
    .strobe {
        border-radius: 8px;
        padding: 14px 16px;
        color: #ffffff;
        background: #b91c1c;
        animation: strobePulse .75s infinite alternate;
        border: 2px solid #fca5a5;
        font-weight: 800;
    }
    @keyframes strobePulse {
        from { box-shadow: 0 0 0 rgba(248,113,113,0); filter: brightness(1); }
        to { box-shadow: 0 0 28px rgba(248,113,113,.85); filter: brightness(1.35); }
    }
    .event-row {
        border-left: 5px solid #64748b;
        padding: 10px 12px;
        margin-bottom: 8px;
        background: rgba(15,23,42,.06);
        border-radius: 6px;
    }
    .event-low { border-left-color: #16a34a; }
    .event-medium { border-left-color: #2563eb; }
    .event-high { border-left-color: #d97706; }
    .event-critical { border-left-color: #dc2626; }
    .muted { color: #64748b; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> ComplianceDatabase:
    db_path = os.environ.get("COMPLIANCE_DB_PATH", "data/compliance_events.db")
    return ComplianceDatabase(db_path)


def css_severity(severity: str) -> str:
    return severity.lower().replace(" ", "-")


def escape(value) -> str:
    return html.escape(str(value if value is not None else ""))


def records_to_csv(records: list[dict]) -> str:
    if not records:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REPORT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def render_event_row(event: dict) -> None:
    report = db.to_report_record(event)
    severity = report["severity"]
    css = css_severity(severity)
    clip_timestamp = event.get("clip_timestamp") or "00:00:00.000"
    st.markdown(
        f"""
        <div class="event-row event-{css}">
            <span class="status-pill status-{css}">{escape(severity)}</span>
            <strong>{escape(report["behavior_class"])}</strong>
            <div class="muted">{escape(report["timestamp"])} | {escape(report["clip_id"])} @ {escape(clip_timestamp)} | {escape(report["zone"])}</div>
            <div>{escape(report["event_description"])}</div>
            <div class="muted">{escape(report["policy_rule_ref"])} | {escape(report["escalation_action"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


db = get_database()

severity_options = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

st.sidebar.title("Filters")
selected_severities = st.sidebar.multiselect(
    "Severity",
    options=severity_options,
    default=severity_options,
)

all_events_sample = db.get_events(limit=1000)
behavior_classes = sorted({event.get("behavior_class", "") for event in all_events_sample if event.get("behavior_class")})
if not behavior_classes:
    behavior_classes = [
        "Safe Walkway Violation",
        "Unauthorized Intervention",
        "Opened Panel Cover",
        "Carrying Overload with Forklift",
    ]

selected_behaviors = st.sidebar.multiselect(
    "Behavior Class",
    options=behavior_classes,
    default=behavior_classes,
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
)

refresh_interval = st.sidebar.slider(
    "Refresh seconds",
    min_value=3,
    max_value=60,
    value=5,
)

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh")
except ImportError:
    st.sidebar.info("Install streamlit-autorefresh to enable timed refresh.")


def filter_events() -> list[dict]:
    start_time = None
    end_time = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_time = datetime.combine(date_range[0], datetime.min.time()).isoformat()
        end_time = datetime.combine(date_range[1], datetime.max.time()).isoformat()

    filtered: list[dict] = []
    for severity in selected_severities:
        for behavior in selected_behaviors:
            filtered.extend(
                db.get_events(
                    severity=severity,
                    behavior_class=behavior,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000,
                )
            )

    seen: set[int] = set()
    unique: list[dict] = []
    for event in filtered:
        row_id = event.get("id")
        if row_id not in seen:
            seen.add(row_id)
            unique.append(event)

    unique.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
    return unique


st.title("Factory Compliance Monitor")
st.caption("Compliance status, strobe alerts, audit timeline, and CSV export")

total_events = db.total_count()
severity_counts = db.count_by_severity()

metric_cols = st.columns(5)
metric_cols[0].metric("Reports", total_events)
metric_cols[1].metric("LOW", severity_counts.get("LOW", 0))
metric_cols[2].metric("MEDIUM", severity_counts.get("MEDIUM", 0))
metric_cols[3].metric("HIGH", severity_counts.get("HIGH", 0))
metric_cols[4].metric("CRITICAL", severity_counts.get("CRITICAL", 0))

alerts = db.get_recent_alerts(count=10)
latest_events = db.get_recent_events(count=25)
latest_event = latest_events[0] if latest_events else None
latest_alert = alerts[0] if alerts else None

st.divider()
st.header("View A - Live Feed Monitor")

if latest_alert:
    current = db.to_report_record(latest_alert)
    current_status = "ALERT ACTIVE"
    current_severity = current["severity"]
elif latest_event:
    current = db.to_report_record(latest_event)
    current_status = "VIOLATION DETECTED"
    current_severity = current["severity"]
else:
    current = {
        "clip_id": "waiting-for-feed",
        "zone": "Zone-1",
        "behavior_class": "No violation detected",
        "event_description": "No compliance events have been recorded.",
        "severity": "CLEAR",
        "timestamp": "-",
    }
    current_status = "NO VIOLATION DETECTED"
    current_severity = "CLEAR"

feed_col, status_col = st.columns([2, 1])
with feed_col:
    color = {
        "LOW": "#16a34a",
        "MEDIUM": "#2563eb",
        "HIGH": "#d97706",
        "CRITICAL": "#dc2626",
        "CLEAR": "#94a3b8",
    }.get(current_severity, "#94a3b8")
    st.markdown(
        f"""
        <div class="feed-panel">
            <div>
                <strong>Processed clip:</strong> {escape(current["clip_id"])}
                <span style="float:right;">{escape(current["zone"])}</span>
            </div>
            <div class="feed-grid" style="color:{color};">
                <div class="feed-zone"></div>
                <div class="feed-target"></div>
            </div>
            <div>{escape(current["behavior_class"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col:
    status_class = "clear" if current_severity == "CLEAR" else css_severity(current_severity)
    st.markdown(
        f"""
        <span class="status-pill status-{status_class}">{escape(current_status)}</span>
        <p><strong>{escape(current.get("severity", ""))}</strong></p>
        <p>{escape(current.get("event_description", ""))}</p>
        <p class="muted">{escape(current.get("timestamp", ""))}</p>
        """,
        unsafe_allow_html=True,
    )

if latest_alert:
    alert_report = db.to_report_record(latest_alert)
    st.markdown(
        f"""
        <div class="strobe">
            DASHBOARD STROBE ALERT: {escape(alert_report["severity"])}
            - {escape(alert_report["behavior_class"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.header("View B - Alert Timeline Stream")

if latest_events:
    for event in latest_events:
        render_event_row(event)
else:
    st.info("No compliance events have been recorded yet.")

st.divider()
st.header("View C - Historical Log & Export")

filtered_events = filter_events()
report_records = [db.to_report_record(event) for event in filtered_events]

if report_records:
    import pandas as pd

    st.dataframe(
        pd.DataFrame(report_records),
        use_container_width=True,
        height=420,
        column_config={
            "event_id": st.column_config.TextColumn("Event ID", width="medium"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "clip_id": st.column_config.TextColumn("Clip", width="small"),
            "zone": st.column_config.TextColumn("Zone", width="small"),
            "behavior_class": st.column_config.TextColumn("Behavior", width="medium"),
            "policy_rule_ref": st.column_config.TextColumn("Policy Ref", width="small"),
            "event_description": st.column_config.TextColumn("Description", width="large"),
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "escalation_action": st.column_config.TextColumn("Escalation", width="medium"),
        },
    )

    csv_data = records_to_csv(report_records)
    st.download_button(
        label="Download filtered audit CSV",
        data=csv_data,
        file_name=f"compliance_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="csv_download",
    )
    st.caption(f"Showing {len(report_records)} report records")
else:
    st.info("No records match the current filters.")