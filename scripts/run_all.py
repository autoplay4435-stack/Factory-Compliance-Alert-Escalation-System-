"""
CLI Script - Run Full System.

Orchestrates all modules:
  1. Parse policy document or use fallback rules.
  2. Initialize SQLite compliance reporting.
  3. Launch Streamlit dashboard.
  4. Run the detection engine through the direct escalation pipeline.

Usage:
    python scripts/run_all.py --video data/factory_clip.mp4
    python scripts/run_all.py --fallback --webcam 0
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

from src.detection.engine import DetectionEngine
from src.escalation.pipeline import EscalationPipeline
from src.models import PolicyRuleSet
from src.reports.database import ComplianceDatabase
from src.severity.parser import PolicyParser, create_fallback_rules


def main() -> None:
    """Run the complete compliance system."""
    parser = argparse.ArgumentParser(
        description="Run the complete Factory Compliance and Alert Escalation System"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video", type=str, help="Path to a video file")
    source_group.add_argument(
        "--webcam", type=int, default=0, help="Webcam index (default: 0)"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="compliance_policy.pdf",
        help="Path to the policy document",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="outputs/policy_rules.json",
        help="Path for the rules JSON",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/compliance_events.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback rules without an LLM call",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip launching the Streamlit dashboard",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated video frames",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=3,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Factory Compliance and Alert Escalation System")
    print("=" * 60)

    rules_path = Path(args.rules)
    if args.fallback or not Path(args.policy).exists():
        print("\nStep 1: Using fallback policy rules...")
        rules = create_fallback_rules()
    elif rules_path.exists():
        print(f"\nStep 1: Loading existing rules from {args.rules}...")
        rules = PolicyRuleSet.from_json_file(str(rules_path))
    else:
        print(f"\nStep 1: Parsing policy document: {args.policy}...")
        try:
            policy_parser = PolicyParser()
            rules = policy_parser.parse_file(args.policy, output_path=str(rules_path))
        except Exception as exc:
            print(f"Policy parsing failed ({exc}); using fallback rules")
            rules = create_fallback_rules()

    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_json_file(str(rules_path))
    print(f"Loaded {len(rules.rules)} rules")

    print(f"\nStep 2: Initializing database: {args.db}")
    database = ComplianceDatabase(args.db)
    pipeline = EscalationPipeline(database)
    print(f"Database ready ({database.total_count()} existing events)")

    dashboard_process = None
    if not args.no_dashboard:
        print("\nStep 3: Launching Streamlit dashboard...")
        dashboard_path = Path(_project_root) / "src" / "dashboard" / "app.py"
        try:
            env = os.environ.copy()
            env["COMPLIANCE_DB_PATH"] = str(Path(args.db).resolve())
            dashboard_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(dashboard_path),
                    "--server.headless",
                    "true",
                ],
                env=env,
                cwd=_project_root,
            )
            print("Dashboard launching at http://localhost:8501")
        except Exception as exc:
            print(f"Failed to launch dashboard: {exc}")
    else:
        print("\nStep 3: Dashboard launch skipped")

    source = args.video if args.video else args.webcam
    print(f"\nStep 4: Starting detection engine on: {source}")
    print("Press ESC (if --display) or Ctrl+C to stop\n")
    print("=" * 60)

    engine = DetectionEngine(
        rules=rules,
        escalation_pipeline=pipeline,
        process_every_n_frames=args.skip_frames,
        display=args.display,
    )

    try:
        engine.run(source=source)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()
        database.close()

        if dashboard_process is not None:
            dashboard_process.terminate()
            print("Dashboard stopped")

    print("\nSystem shutdown complete")
    print(f"Total events routed this run: {pipeline.event_count}")
    print(f"Dashboard strobe alerts this run: {pipeline.alert_count}")
    print(f"Database: {args.db}")


if __name__ == "__main__":
    main()
