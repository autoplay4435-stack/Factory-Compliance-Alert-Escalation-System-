"""
CLI Script - Run Detection Engine.

Starts the detection engine on a video source with policy rules loaded.

Usage:
    python scripts/run_detection.py --video data/factory_clip.mp4 --rules outputs/policy_rules.json
    python scripts/run_detection.py --webcam 0 --rules outputs/policy_rules.json --display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

from src.detection.engine import DetectionEngine
from src.escalation.pipeline import EscalationPipeline
from src.models import PolicyRuleSet
from src.reports.database import ComplianceDatabase
from src.severity.parser import create_fallback_rules


def main() -> None:
    """Run the detection engine."""
    parser = argparse.ArgumentParser(
        description="Run the factory compliance detection engine"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video", type=str, help="Path to a video file")
    source_group.add_argument(
        "--webcam", type=int, default=0, help="Webcam index (default: 0)"
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="outputs/policy_rules.json",
        help="Path to policy rules JSON",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/compliance_events.db",
        help="Path to SQLite database",
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

    rules_path = Path(args.rules)
    if rules_path.exists():
        print(f"Loading rules from: {args.rules}")
        rules = PolicyRuleSet.from_json_file(str(rules_path))
    else:
        print(f"Rules file not found ({args.rules}), using fallback rules")
        rules = create_fallback_rules()

    print(f"Loaded {len(rules.rules)} policy rules")

    database = ComplianceDatabase(args.db)
    pipeline = EscalationPipeline(database)
    print(f"Database initialized: {args.db}")

    source = args.video if args.video else args.webcam
    engine = DetectionEngine(
        rules=rules,
        escalation_pipeline=pipeline,
        process_every_n_frames=args.skip_frames,
        display=args.display,
    )

    print(f"\nStarting detection on: {source}")
    print("Press ESC (if --display) or Ctrl+C to stop\n")

    try:
        engine.run(source=source)
    except KeyboardInterrupt:
        print("\nStopping detection engine...")
        engine.stop()
    finally:
        database.close()

    print(f"\nDetection complete. Events stored in: {args.db}")
    print(f"Total events routed this run: {pipeline.event_count}")
    print(f"Dashboard strobe alerts this run: {pipeline.alert_count}")
    print(f"Total events in database: {database.total_count()}")


if __name__ == "__main__":
    main()
