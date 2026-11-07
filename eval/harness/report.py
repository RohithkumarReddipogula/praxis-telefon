"""Summarise one harness run: rates with intervals, latency percentiles, failures, review.

    uv run python -m eval.harness.report eval/results/runs/<stamp>

Rates: per scenario with Wilson 95 % intervals (5 runs: wide, e.g. 5/5 = 57-100 %);
overall pooled over conversations with a bootstrap over scenarios, never narrower than
Wilson on the pooled count. Latency per caller
turn (text mode: time to first speech and to the answer), p50 / p95 with a bootstrap over
conversations. Conversations that errored are counted, not scored. Free: no API calls.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.harness import stats

METRICS = ("task_success", "tools_ok", "no_unbacked_claim", "no_invented_fact", "right_language")
LATENCIES = ("first_speech_s", "first_answer_s")


@dataclass
class Run:
    path: Path
    meta: dict[str, Any]
    conversations: list[dict[str, Any]]

    def scored(self) -> list[dict[str, Any]]:
        return [c for c in self.conversations if c.get("error") is None and c.get("metrics")]


def load(path: Path) -> Run:
    lines = (path / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Run(path, meta, [json.loads(line) for line in lines if line.strip()])


def clusters(run: Run, metric: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for c in run.scored():
        out[c["scenario"]].append(int(bool(c["metrics"][metric])))
    return dict(out)


def latencies(run: Run, key: str) -> list[list[float]]:
    """Per conversation, the turn latencies that were measured."""
    return [[t[key] for t in c.get("turns", []) if t.get(key) is not None] for c in run.scored()]


def _rate(values: list[int]) -> dict[str, Any]:
    k, n = sum(values), len(values)
    lo, hi = stats.wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def summarise(run: Run) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    per_scenario: dict[str, dict[str, Any]] = defaultdict(dict)
    for metric in METRICS:
        by = clusters(run, metric)
        pooled = [v for vs in by.values() for v in vs]
        lo, hi = stats.cluster_bootstrap_rate(list(by.values()))
        # Never narrower than Wilson on the pooled count: with all successes (or all
        # failures) resampling scenarios cannot vary (smoke run 2026-11-07: 100-100 %).
        rate = _rate(pooled)
        overall[metric] = {**rate, "lo": min(lo, rate["lo"]), "hi": max(hi, rate["hi"])}
        for sid, vs in by.items():
            per_scenario[sid][metric] = _rate(vs)
    latency = {}
    for key in LATENCIES:
        convs = latencies(run, key)
        flat = [x for c in convs for x in c]
        latency[key] = {
            "n": len(flat),
            "p50": stats.percentile(flat, 0.5),
            "p50_ci": stats.bootstrap_percentile(convs, 0.5),
            "p95": stats.percentile(flat, 0.95),
            "p95_ci": stats.bootstrap_percentile(convs, 0.95),
        }
    failures = [
        (c["scenario"], c["run"], c["metrics"]["failures"])
        for c in run.scored()
        if c["metrics"]["failures"]
    ]
    review = [
        (c["scenario"], c["run"], text) for c in run.scored() for text in c["metrics"]["review"]
    ]
    return {
        "meta": run.meta,
        "conversations": len(run.conversations),
        "errors": sum(c.get("error") is not None for c in run.conversations),
        "cost_usd": round(sum(c.get("cost_usd") or 0 for c in run.conversations), 4),
        "overall": overall,
        "per_scenario": dict(per_scenario),
        "latency": latency,
        "failures": failures,
        "review": review,
    }


def _pct(x: float) -> str:
    return f"{100 * x:5.1f} %"


def render(s: dict[str, Any]) -> str:
    m = s["meta"]
    lines = [
        f"run {m.get('git_commit', '?')}, model {m.get('model', '?')}, "
        f"{s['conversations']} conversations, {s['errors']} errors, ${s['cost_usd']:.3f}"
        + (" (STOPPED AT COST CAP)" if m.get("stopped_at_cap") else ""),
        "",
        "Overall (95 % interval, bootstrap over scenarios):",
    ]
    for metric, r in s["overall"].items():
        interval = f"{_pct(r['lo'])} - {_pct(r['hi'])}"
        lines.append(f"  {metric:18} {_pct(r['rate'])}  ({r['k']}/{r['n']}, {interval})")
    lines.append("\nLatency per caller turn (s, 95 % interval):")
    for key, r in s["latency"].items():
        lines.append(
            f"  {key:15} p50 {r['p50']:.2f} ({r['p50_ci'][0]:.2f}-{r['p50_ci'][1]:.2f})  "
            f"p95 {r['p95']:.2f} ({r['p95_ci'][0]:.2f}-{r['p95_ci'][1]:.2f})  n={r['n']}"
        )
    lines.append("\nPer scenario, task success (Wilson 95 %):")
    for sid, r in sorted(s["per_scenario"].items()):
        t = r["task_success"]
        lines.append(f"  {sid:32} {t['k']}/{t['n']}  ({_pct(t['lo'])} - {_pct(t['hi'])})")
    if s["failures"]:
        lines.append("\nFailed checks:")
        lines += [f"  {sid} [{run}]: {', '.join(f)}" for sid, run, f in s["failures"]]
    if s["review"]:
        lines.append("\nReplies to review (invented facts?):")
        lines += [f"  {sid} [{run}]: {text!r}" for sid, run, text in s["review"]]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    print(render(summarise(load(Path(argv[0])))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
