"""
CLI Script — Parse Policy Document.

Reads a factory policy document, extracts compliance rules using the LLM,
and writes the structured output to a JSON file.

Usage:
    python scripts/run_parser.py --policy compliance_policy.pdf --output outputs/policy_rules.json
    python scripts/run_parser.py --fallback  # Use hardcoded fallback rules (no API key needed)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

from src.severity.parser import PolicyParser, create_fallback_rules


def main() -> None:
    """Parse a policy document and output structured rules."""
    parser = argparse.ArgumentParser(
        description="Parse a factory policy document into structured compliance rules"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="compliance_policy.pdf",
        help="Path to the policy document (default: compliance_policy.pdf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/policy_rules.json",
        help="Path for the output JSON (default: outputs/policy_rules.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use hardcoded fallback rules instead of calling the LLM",
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

    if args.fallback:
        print("Using hardcoded fallback rules (no LLM call)...")
        rule_set = create_fallback_rules()
    else:
        print(f"Parsing policy document: {args.policy}")
        policy_parser = PolicyParser(model=args.model)
        rule_set = policy_parser.parse_file(args.policy, output_path=None)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    rule_set.to_json_file(str(output_path))
    print(f"\n✅ Successfully extracted {len(rule_set.rules)} rules")
    print(f"   Output written to: {args.output}")

    # Print summary
    print("\n--- Extracted Rules ---")
    for i, rule in enumerate(rule_set.rules, 1):
        print(
            f"  {i}. [{rule.assigned_severity.value}] {rule.behavior_class} "
            f"(ref: {rule.policy_section_ref}, type: {rule.detection_type.value})"
        )


if __name__ == "__main__":
    main()
