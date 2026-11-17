"""Automated evaluation suite verifying agent trajectory, policy compliance, and guardrails."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.graph import resolve_incident  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.database.mock_store import reset_store  # noqa: E402
from src.schemas import IncidentResolveRequest  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run a single evaluation scenario and compute trajectory/policy/guardrail checks."""
    request = IncidentResolveRequest(
        order_id=case["order_id"],
        customer_message=case["customer_message"],
    )
    response = resolve_incident(request)
    settings = get_settings()

    actual_tools = [tool.tool_name for tool in response.telemetry.tools_executed]
    trajectory_ok = actual_tools == case["expected_tools"]

    policy_ok = (
        response.decision.action_type == case["expected_action"]
        and math.isclose(response.decision.amount, float(case["expected_amount"]), abs_tol=0.01)
        and response.decision.policy_reference == case["expected_policy_reference"]
    )

    within_hard_limits = (
        response.decision.amount <= settings.max_auto_refund_brl
        and (
            response.decision.action_type != "ISSUE_COUPON"
            or response.decision.amount <= settings.max_auto_coupon_brl
        )
    )
    guardrail_ok = (
        response.decision.guardrail_overridden is bool(case["expected_guardrail_overridden"])
        and within_hard_limits
    )

    passed = trajectory_ok and policy_ok and guardrail_ok
    return {
        "id": case["id"],
        "order_id": case["order_id"],
        "expected_action": case["expected_action"],
        "actual_action": response.decision.action_type,
        "amount": response.decision.amount,
        "guardrail": response.decision.guardrail_overridden,
        "trajectory_ok": trajectory_ok,
        "policy_ok": policy_ok,
        "guardrail_ok": guardrail_ok,
        "latency_ms": response.telemetry.latency_ms,
        "tokens": response.telemetry.input_tokens + response.telemetry.output_tokens,
        "cost_usd": response.telemetry.estimated_cost_usd,
        "status": "PASS" if passed else "FAIL",
    }


def _print_report(results: list[dict[str, Any]]) -> None:
    """Print an ASCII/UTF-8 evaluation summary table to stdout."""
    headers = (
        f"{'ID':<8} | {'Order':<8} | {'Expected':<15} | {'Actual':<15} | "
        f"{'Amount':>8} | {'Guard':<5} | {'Traj':<4} | {'Pol':<4} | {'Latency':>9} | {'Result':<6}"
    )
    separator = "-" * len(headers)
    print("\n" + separator)
    print("iFood Incident Triage Agent — Automated Eval Suite")
    print(separator)
    print(headers)
    print(separator)

    for row in results:
        print(
            f"{row['id']:<8} | {row['order_id']:<8} | {row['expected_action']:<15} | "
            f"{row['actual_action']:<15} | R$ {row['amount']:>5.2f} | "
            f"{str(row['guardrail']):<5} | "
            f"{'OK' if row['trajectory_ok'] else 'ERR':<4} | "
            f"{'OK' if row['policy_ok'] else 'ERR':<4} | "
            f"{row['latency_ms']:>7.2f}ms | {row['status']:<6}"
        )

    print(separator)
    passed_count = sum(1 for row in results if row["status"] == "PASS")
    total_count = len(results)
    avg_latency = (sum(row["latency_ms"] for row in results) / total_count) if total_count else 0.0
    total_tokens = sum(row["tokens"] for row in results)
    total_cost = sum(row["cost_usd"] for row in results)
    pass_rate = (passed_count / total_count * 100.0) if total_count else 0.0

    print(
        f"Summary: {passed_count}/{total_count} passed ({pass_rate:.1f}%) | "
        f"Avg Latency: {avg_latency:.2f} ms | Total Tokens: {total_tokens} | "
        f"Est. Cost: ${total_cost:.6f} USD\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated triage agent evaluations.")
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        default="mock",
        help="LLM provider to use during evaluation (default: mock for deterministic CI).",
    )
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = args.provider
    get_settings.cache_clear()
    reset_store()

    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        dataset: list[dict[str, Any]] = json.load(fh)

    results = [_evaluate_case(case) for case in dataset]
    _print_report(results)

    all_passed = all(row["status"] == "PASS" for row in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
