"""Compare two evaluation summaries with paired task level statistics.

The script reads the JSON produced by evaluate.py for Arm A and Arm B, aligns
tasks, reports per task differences, and estimates a paired bootstrap interval
for the median speedup difference.
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def task_speedups(summary: Dict[str, Any], split: str = "all") -> Dict[str, float]:
    if split == "all":
        records = summary.get("task_best", summary.get("tasks", []))
    else:
        records = summary.get("by_split", {}).get(split, {}).get("task_best", [])
    return {
        record["task_id"]: float(record.get("speedup", 0.0))
        for record in records
        if "task_id" in record
    }


def task_correctness(summary: Dict[str, Any], split: str = "all") -> Dict[str, bool]:
    if split == "all":
        records = summary.get("task_best", summary.get("tasks", []))
    else:
        records = summary.get("by_split", {}).get(split, {}).get("task_best", [])
    return {
        record["task_id"]: bool(record.get("numerically_correct", False))
        for record in records
        if "task_id" in record
    }


def bootstrap_median_ci(values: List[float], samples: int, seed: int) -> List[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        resample = [rng.choice(values) for _ in values]
        medians.append(statistics.median(resample))
    medians.sort()
    lower = medians[int(0.025 * len(medians))]
    upper = medians[int(0.975 * len(medians)) - 1]
    return [round(lower, 4), round(upper, 4)]


def compare(
    arm_a: Dict[str, Any],
    arm_b: Dict[str, Any],
    split: str = "all",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    speedups_a = task_speedups(arm_a, split)
    speedups_b = task_speedups(arm_b, split)
    correctness_a = task_correctness(arm_a, split)
    correctness_b = task_correctness(arm_b, split)

    shared_tasks = sorted(set(speedups_a) & set(speedups_b))
    differences = [speedups_a[task] - speedups_b[task] for task in shared_tasks]
    correctness_differences = [
        int(correctness_a.get(task, False)) - int(correctness_b.get(task, False))
        for task in shared_tasks
    ]

    per_task = []
    wins = ties = losses = 0
    for task in shared_tasks:
        difference = speedups_a[task] - speedups_b[task]
        if difference > 0.005:
            wins += 1
        elif difference < -0.005:
            losses += 1
        else:
            ties += 1
        per_task.append(
            {
                "task_id": task,
                "arm_a_speedup": round(speedups_a[task], 4),
                "arm_b_speedup": round(speedups_b[task], 4),
                "difference": round(difference, 4),
                "arm_a_correct": correctness_a.get(task, False),
                "arm_b_correct": correctness_b.get(task, False),
            }
        )

    observed = statistics.median(differences) if differences else 0.0
    return {
        "split": split,
        "num_tasks": len(shared_tasks),
        "arm_a_median_speedup": round(statistics.median(list(speedups_a.values())), 4)
        if speedups_a
        else 0.0,
        "arm_b_median_speedup": round(statistics.median(list(speedups_b.values())), 4)
        if speedups_b
        else 0.0,
        "median_difference": round(observed, 4),
        "median_difference_ci": bootstrap_median_ci(differences, bootstrap_samples, seed),
        "mean_difference": round(statistics.mean(differences), 4) if differences else 0.0,
        "wins_a": wins,
        "ties": ties,
        "wins_b": losses,
        "arm_a_correctness_rate": round(
            sum(1 for task in shared_tasks if correctness_a.get(task, False)) / len(shared_tasks),
            4,
        )
        if shared_tasks
        else 0.0,
        "arm_b_correctness_rate": round(
            sum(1 for task in shared_tasks if correctness_b.get(task, False)) / len(shared_tasks),
            4,
        )
        if shared_tasks
        else 0.0,
        "correctness_difference": round(
            sum(correctness_differences) / len(correctness_differences), 4
        )
        if correctness_differences
        else 0.0,
        "per_task": per_task,
    }


def print_comparison(result: Dict[str, Any]) -> None:
    print(f"Split: {result['split']}")
    print(f"Shared tasks: {result['num_tasks']}")
    print(f"Arm A median speedup: {result['arm_a_median_speedup']:.4f}x")
    print(f"Arm B median speedup: {result['arm_b_median_speedup']:.4f}x")
    print(
        f"Paired median difference: {result['median_difference']:+.4f}x "
        f"CI {result['median_difference_ci']}"
    )
    print(
        f"Correctness: A={result['arm_a_correctness_rate']:.2%} "
        f"B={result['arm_b_correctness_rate']:.2%} "
        f"difference={result['correctness_difference']:+.2%}"
    )
    print(f"Task wins: A={result['wins_a']} ties={result['ties']} B={result['wins_b']}")
    print()
    print(f"{'task':<24}{'arm_a':>10}{'arm_b':>10}{'diff':>10}")
    for record in result["per_task"]:
        print(
            f"{record['task_id']:<24}"
            f"{record['arm_a_speedup']:>10.4f}"
            f"{record['arm_b_speedup']:>10.4f}"
            f"{record['difference']:>+10.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Arm A and Arm B evaluations.")
    parser.add_argument("--arm_a", required=True, help="Arm A summary JSON.")
    parser.add_argument("--arm_b", required=True, help="Arm B summary JSON.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "training", "holdout"],
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    summary_a = load_summary(args.arm_a)
    summary_b = load_summary(args.arm_b)
    result = compare(
        summary_a,
        summary_b,
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print_comparison(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nWrote comparison to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
