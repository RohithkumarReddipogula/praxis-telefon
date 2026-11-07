"""M4 run report and run comparison, on made-up conversation records (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.harness import compare, report

METRICS = ("task_success", "tools_ok", "no_unbacked_claim", "no_invented_fact", "right_language")


def _conv(scenario: str, run: int, ok: bool, first_speech: float = 1.0) -> dict[str, Any]:
    metrics = {m: ok for m in METRICS} | {"turns": 2, "failures": [] if ok else ["x"], "review": []}
    return {
        "scenario": scenario,
        "run": run,
        "error": None,
        "cost_usd": 0.002,
        "turns": [{"first_speech_s": first_speech, "first_answer_s": first_speech + 0.8}],
        "metrics": metrics,
        "group": "appointments",
    }


def _write(tmp: Path, name: str, convs: list[dict[str, Any]]) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "conversations.jsonl").write_text("".join(json.dumps(c) + "\n" for c in convs))
    (d / "meta.json").write_text(
        json.dumps({"git_commit": name, "model": "m", "spent_usd": 0.4, "stopped_at_cap": False})
    )
    return d


def test_summary_rates_intervals_and_latency(tmp_path: Path) -> None:
    convs = [_conv(f"s{i}", r, ok=(i != 0 or r < 3)) for i in range(10) for r in range(5)]
    s = report.summarise(report.load(_write(tmp_path, "a", convs)))
    ts = s["overall"]["task_success"]
    assert ts["k"] == 48 and ts["n"] == 50
    assert ts["lo"] < 0.96 < ts["hi"]
    assert s["per_scenario"]["s0"]["task_success"]["k"] == 3
    assert s["latency"]["first_speech_s"]["p50"] == 1.0
    assert s["errors"] == 0


def test_errors_are_counted_not_scored(tmp_path: Path) -> None:
    convs = [
        _conv("s0", 0, ok=True),
        {**_conv("s0", 1, ok=True), "error": "Timeout", "metrics": None},
    ]
    s = report.summarise(report.load(_write(tmp_path, "a", convs)))
    assert s["errors"] == 1 and s["overall"]["task_success"]["n"] == 1


def test_render_lists_failures_and_review(tmp_path: Path) -> None:
    bad = _conv("s1", 0, ok=False)
    bad["metrics"]["review"] = ["Dazu weiß ich nichts."]
    text = report.render(report.summarise(report.load(_write(tmp_path, "a", [bad]))))
    assert "s1" in text and "Dazu weiß ich nichts." in text


def test_comparison_flags_a_clear_regression(tmp_path: Path) -> None:
    good = [_conv(f"s{i}", r, ok=True) for i in range(10) for r in range(5)]
    bad = [_conv(f"s{i}", r, ok=(i >= 6)) for i in range(10) for r in range(5)]
    c = compare.compare(
        report.load(_write(tmp_path, "a", good)), report.load(_write(tmp_path, "b", bad))
    )
    ts = c["metrics"]["task_success"]
    assert ts["verdict"] == "hurts" and ts["diff_hi"] < 0
    assert {r["scenario"] for r in c["worse_scenarios"]} == {"s0", "s1", "s2", "s3", "s4", "s5"}


def test_comparison_of_equal_runs_shows_no_change(tmp_path: Path) -> None:
    runs = [_conv(f"s{i}", r, ok=(r != 0)) for i in range(10) for r in range(5)]
    c = compare.compare(
        report.load(_write(tmp_path, "a", runs)), report.load(_write(tmp_path, "b", runs))
    )
    assert c["metrics"]["task_success"]["verdict"] == "no clear change"
    assert c["worse_scenarios"] == []


def test_comparison_of_latency(tmp_path: Path) -> None:
    fast = [_conv(f"s{i}", r, ok=True, first_speech=1.0) for i in range(10) for r in range(5)]
    slow = [_conv(f"s{i}", r, ok=True, first_speech=1.5) for i in range(10) for r in range(5)]
    c = compare.compare(
        report.load(_write(tmp_path, "a", fast)), report.load(_write(tmp_path, "b", slow))
    )
    lat = c["latency"]["first_speech_s"]["p50"]
    assert abs(lat["diff"] - 0.5) < 1e-9 and lat["verdict"] == "hurts"


def test_comparison_warns_about_different_settings(tmp_path: Path) -> None:
    a = report.load(_write(tmp_path, "a", [_conv("s0", 0, ok=True)]))
    b = report.load(_write(tmp_path, "b", [_conv("s0", 0, ok=True)]))
    b.meta["model"] = "other"
    assert any("model" in w for w in compare.compare(a, b)["warnings"])


def test_all_successes_do_not_give_a_zero_width_interval(tmp_path: Path) -> None:
    # Smoke run 2026-11-07: 2/2 successes gave the overall interval 100-100 %; resampling
    # scenarios cannot vary when every one succeeded. The interval is at least as wide as
    # Wilson's on the pooled count.
    convs = [_conv(f"s{i}", 0, ok=True) for i in range(2)]
    ts = report.summarise(report.load(_write(tmp_path, "a", convs)))["overall"]["task_success"]
    assert ts["lo"] < 0.5 and ts["hi"] == 1.0


def test_latency_is_compared_on_the_scenarios_both_runs_have(tmp_path: Path) -> None:
    # M4 rerun 2026-11-07: run B had 6 new scenarios; pooling all latencies compared
    # different scenario mixes. A slow new scenario must not change the comparison.
    a = [_conv(f"s{i}", r, ok=True, first_speech=1.0) for i in range(10) for r in range(5)]
    b = a + [_conv("new", r, ok=True, first_speech=9.0) for r in range(5)]
    c = compare.compare(
        report.load(_write(tmp_path, "a", a)), report.load(_write(tmp_path, "b", b))
    )
    assert c["latency"]["first_speech_s"]["p95"]["diff"] == 0.0
    assert any("only in" in w for w in c["warnings"])
