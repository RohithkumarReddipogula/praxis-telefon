"""Report the STT A/B: accuracy per setting, the decision rule for multi, detection rates.

    uv run python -m eval.stt_ab.report

Reads eval/results/stt_ab/transcripts.jsonl (run_stt.py) and the utterances. Pure
computation; no API calls.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval.stt_ab.run_stt import RESULTS
from eval.stt_ab.score import item_correct, mcnemar_p, multi_decision, wer
from eval.stt_ab.utterances import Utterance, load
from praxis_telefon.agent.speech_filters import detect_language

KINDS = ("name", "dob", "phone")


def scored(records: Iterable[dict[str, Any]], utterances: list[Utterance]) -> list[dict[str, Any]]:
    """Each transcript with its utterance, right/wrong (names folded and strict), WER."""
    by_id = {u.id: u for u in utterances}
    out = []
    for r in records:
        u = by_id[r["id"]]
        reference = " ".join(s.text for s in u.segments)
        out.append({
            **r,
            "group": u.group,
            "kind": u.kind,
            "right": item_correct(u.kind, u.expected, r["text"], folded=True),
            "right_strict": item_correct(u.kind, u.expected, r["text"], folded=False),
            "wer": wer(reference, r["text"]),
            "detected": detect_language(r["text"]),
        })  # fmt: skip
    return out


def _paired(
    rows: list[dict[str, Any]], setting: str, base: str = "de"
) -> tuple[list[bool], list[bool]]:
    """Right/wrong for the same clips under ``base`` and ``setting``."""
    a = {r["clip"]: r["right"] for r in rows if r["setting"] == base}
    b = {r["clip"]: r["right"] for r in rows if r["setting"] == setting}
    clips = sorted(set(a) & set(b))
    return [a[c] for c in clips], [b[c] for c in clips]


def _acc(values: list[bool]) -> str:
    return (
        f"{100 * sum(values) / len(values):5.1f}% ({sum(values)}/{len(values)})"
        if values
        else "  n/a"
    )


def render(rows: list[dict[str, Any]]) -> str:
    settings = sorted({r["setting"] for r in rows}, key=["de", "multi", "multi_hint", "en"].index)
    lines = ["Item accuracy (names folded; strict in brackets for names):"]
    for group in ("de", "en", "switch"):
        for setting in settings:
            sel = [r for r in rows if r["setting"] == setting and r["group"] == group]
            if not sel:
                continue
            parts = []
            for kind in KINDS:
                k = [r for r in sel if r["kind"] == kind]
                text = _acc([r["right"] for r in k])
                if kind == "name":
                    text += f" [{_acc([r['right_strict'] for r in k]).split()[0]}]"
                parts.append(f"{kind} {text}")
            for condition in ("clean", "phone"):
                c = [r["right"] for r in sel if r["condition"] == condition]
                parts.append(f"{condition} {_acc(c).split()[0]}")
            mean_wer = statistics.mean(r["wer"] for r in sel)
            lines.append(
                f"  {group:6} {setting:10} " + " | ".join(parts) + f" | WER {mean_wer:.2f}"
            )
    lines.append("\nDecision rule on German clips (combined drop > 2 points rejects):")
    for setting in (s for s in settings if s.startswith("multi")):
        de_k: dict[str, list[bool]] = defaultdict(list)
        mu_k: dict[str, list[bool]] = defaultdict(list)
        for kind in KINDS:
            sel = [r for r in rows if r["group"] == "de" and r["kind"] == kind]
            de_k[kind], mu_k[kind] = _paired(sel, setting)
        lines.append(f"  {setting}: {multi_decision(dict(de_k), dict(mu_k))}")
    lines.append("\nOther settings vs de on the same clips (McNemar p):")
    for setting in (s for s in settings if s != "de"):
        for group in ("en", "switch"):
            a, b = _paired([r for r in rows if r["group"] == group], setting)
            if a:
                lines.append(
                    f"  {group:6} {setting:10} de {_acc(a)} vs {_acc(b)}  p={mcnemar_p(a, b):.3f}"
                )
    lines.append(
        "\nLanguage seen in the transcript (detect_language), for 'de, then switch to en':"
    )
    for setting in settings:
        for group in ("de", "en", "switch"):
            sel = [r for r in rows if r["setting"] == setting and r["group"] == group]
            if sel:
                en = sum(r["detected"] == "en" for r in sel)
                lines.append(f"  {setting:10} {group:6} English detected in {en}/{len(sel)}")
    lines.append("\nDeepgram's language label (multi settings), share of finals:")
    for setting in (s for s in settings if s.startswith("multi")):
        for group in ("de", "en"):
            labels = [lang for r in rows if r["setting"] == setting and r["group"] == group
                      for lang in r["languages"]]  # fmt: skip
            if labels:
                de_share = sum(lang.startswith("de") for lang in labels) / len(labels)
                lines.append(
                    f"  {setting:10} {group}-clips: de {de_share:.0%} of {len(labels)} finals"
                )
    lines.append("\nFinal transcript after the audio ends (s), p50 / p90:")
    for setting in settings:
        delays = sorted(
            r["final_delay_s"]
            for r in rows
            if r["setting"] == setting and r["final_delay_s"] is not None
        )
        if delays:
            p90 = delays[min(len(delays) - 1, int(0.9 * len(delays)))]
            lines.append(f"  {setting:10} {statistics.median(delays):.2f} / {p90:.2f}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else RESULTS
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(render(scored(records, load())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
