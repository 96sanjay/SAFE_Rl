from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best feasible temperature checkpoint.")
    parser.add_argument("--eval-json", type=Path, required=True, help="Evaluation JSON from case-study script.")
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        default=0.05,
        help="Hard feasibility threshold on violation rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.eval_json, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]
    feasible = [r for r in rows if r["name"].lower().startswith("csac-lb") and float(r["violation_rate"]) <= args.max_violation_rate]

    if not feasible:
        print("No feasible CSAC-LB checkpoint found under the requested violation threshold.")
        return

    def score(row: dict) -> float:
        return (
            float(row.get("kpi::cost_total", float("inf")))
            + float(row.get("kpi::carbon_emissions_total", float("inf")))
            + float(row.get("kpi::electricity_consumption_total", float("inf")))
            + float(row.get("kpi::discomfort_proportion", float("inf")))
        )

    best = min(feasible, key=score)
    print("Feasible candidates:")
    for row in feasible:
        print(
            row["name"],
            "violation_rate=", row["violation_rate"],
            "score=", score(row),
            "cost_total=", row.get("kpi::cost_total"),
            "carbon=", row.get("kpi::carbon_emissions_total"),
            "electricity=", row.get("kpi::electricity_consumption_total"),
            "discomfort=", row.get("kpi::discomfort_proportion"),
        )
    print("\nSelected:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
