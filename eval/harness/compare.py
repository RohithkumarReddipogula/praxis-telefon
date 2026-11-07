"""Compare two harness runs: does a change help or hurt?

    uv run python -m eval.harness.compare eval/results/runs/<A> eval/results/runs/<B>

Per metric: A, B and B - A with a 95 % bootstrap interval, resampling scenarios (the same
scenario ids in both runs, so the difference is paired by scenario). "helps" / "hurts"
only when the interval excludes 0; with five metrics, a Bonferroni note says how many
such calls to expect by chance. Latency: the difference of p50 and p95, bootstrapped over
conversations. Worse scenarios: task success down by 2 or more runs, listed even when
not significant. Settings that differ (model, reasoning, prompt, git commit) are warned
about. Free: no API calls.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from eval.harness import stats
from eval.harness.report import LATENCIES, METRICS, Run, clusters, latencies, load

SAMPLES = 2000
SETTINGS = ("model", "reasoning_effort", "prompt_sha", "interruption", "endpointing")


def _verdict(lo: float, hi: float, higher_is_better: bool = True) -> str:
    if lo > 0:
        return "helps" if higher_is_better else "hurts"
    if hi < 0:
        return "hurts" if higher_is_better else "helps"
    return "no clear change"


def _rate_diff(a: dict[str, list[int]], b: dict[str, list[int]], seed: int) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)

    def pooled(by: dict[str, list[int]], pick: list[str]) -> float:
        values = [v for sid in pick for v in by[sid]]
        return sum(values) / len(values) if values else float("nan")

    diffs = []
    for _ in range(SAMPLES):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(pooled(b, pick) - pooled(a, pick))
    lo, hi = stats.percentile(diffs, 0.025), stats.percentile(diffs, 0.975)
    return {
        "a": pooled(a, ids),
        "b": pooled(b, ids),
        "diff": pooled(b, ids) - pooled(a, ids),
        "diff_lo": lo,
        "diff_hi": hi,
        "verdict": _verdict(lo, hi),
    }


def _latency_diff(
    a: list[list[float]], b: list[list[float]], q: float, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)

    def pct(convs: list[list[float]]) -> float:
        return stats.percentile([x for c in convs for x in c], q)

    fa, fb = [c for c in a if c], [c for c in b if c]
    diffs = []
    for _ in range(SAMPLES):
        ra = [fa[rng.randrange(len(fa))] for _ in fa]
        rb = [fb[rng.randrange(len(fb))] for _ in fb]
        diffs.append(pct(rb) - pct(ra))
    lo, hi = stats.percentile(diffs, 0.025), stats.percentile(diffs, 0.975)
    return {
        "a": pct(fa),
        "b": pct(fb),
        "diff": pct(fb) - pct(fa),
        "diff_lo": lo,
        "diff_hi": hi,
        "verdict": _verdict(lo, hi, higher_is_better=False),
    }


def _common(run: Run, ids: set[str]) -> Run:
    """The run restricted to the given scenarios (compare like with like)."""
    return Run(run.path, run.meta, [c for c in run.conversations if c["scenario"] in ids])


def compare(a: Run, b: Run, seed: int = 0) -> dict[str, Any]:
    warnings = [
        f"{key} differs: {a.meta.get(key)!r} vs {b.meta.get(key)!r}"
        for key in SETTINGS
        if a.meta.get(key) != b.meta.get(key)
    ]
    ids_a = {c["scenario"] for c in a.conversations}
    ids_b = {c["scenario"] for c in b.conversations}
    # M4 rerun 2026-11-07: latency was pooled over different scenario mixes.
    for name, only in (("A", ids_a - ids_b), ("B", ids_b - ids_a)):
        if only:
            warnings.append(f"{len(only)} scenarios only in {name}, left out: {sorted(only)}")
    a, b = _common(a, ids_a & ids_b), _common(b, ids_a & ids_b)
    metrics = {m: _rate_diff(clusters(a, m), clusters(b, m), seed) for m in METRICS}
    latency = {
        key: {
            "p50": _latency_diff(latencies(a, key), latencies(b, key), 0.5, seed),
            "p95": _latency_diff(latencies(a, key), latencies(b, key), 0.95, seed),
        }
        for key in LATENCIES
    }
    ta, tb = clusters(a, "task_success"), clusters(b, "task_success")
    worse = [
        {
            "scenario": sid,
            "a": f"{sum(ta[sid])}/{len(ta[sid])}",
            "b": f"{sum(tb[sid])}/{len(tb[sid])}",
        }
        for sid in sorted(set(ta) & set(tb))
        if sum(ta[sid]) - sum(tb[sid]) >= 2
    ]
    calls = sum(r["verdict"] != "no clear change" for r in metrics.values())
    return {
        "warnings": warnings,
        "metrics": metrics,
        "latency": latency,
        "worse_scenarios": worse,
        "significant_calls": calls,
    }


def render(c: dict[str, Any]) -> str:
    lines = [f"WARNING: {w}" for w in c["warnings"]]
    lines.append("Metric               A        B        B-A (95 %)               verdict")
    for m, r in c["metrics"].items():
        lines.append(
            f"  {m:18} {100 * r['a']:5.1f} %  {100 * r['b']:5.1f} %  {100 * r['diff']:+5.1f} "
            f"({100 * r['diff_lo']:+5.1f} .. {100 * r['diff_hi']:+5.1f})  {r['verdict']}"
        )
    lines.append(
        f"  ({c['significant_calls']} of {len(c['metrics'])} metrics called; at a 5 % level "
        f"about {0.05 * len(c['metrics']):.2f} would be called by chance; Bonferroni: "
        f"use 1 % per metric for a 5 % family-wise level)"
    )
    for key, qs in c["latency"].items():
        for q, r in qs.items():
            lines.append(
                f"  {key} {q}: {r['a']:.2f} -> {r['b']:.2f} s ({r['diff']:+.2f}, "
                f"{r['diff_lo']:+.2f} .. {r['diff_hi']:+.2f})  {r['verdict']}"
            )
    if c["worse_scenarios"]:
        lines.append("Scenarios with task success down by 2+ runs:")
        lines += [f"  {w['scenario']}: {w['a']} -> {w['b']}" for w in c["worse_scenarios"]]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    print(render(compare(load(Path(argv[0])), load(Path(argv[1])))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
