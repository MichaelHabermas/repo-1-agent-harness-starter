#!/usr/bin/env python3
"""
Same question, different paths.

Runs one fixed Northwind question once with MockModel, then N times with
OpenRouterModel (when OPENROUTER_API_KEY is set). Writes every trace to
traces/, then prints a compact comparison table.

The teaching point: identical input, different tool sequences, different
answers, different costs. That is why evals exist — you cannot assert exact
outputs once the brain is a real model.

    python3 demo_variance.py
    python3 demo_variance.py -n 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on sys.path so `agent` imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import db, guardrails as guards, trace as tracing
from agent.harness import Agent, AgentConfig
from agent.model import MockModel, OpenRouterModel, load_dotenv

# Poisoned-handbook case: mock path follows the injection; live paths
# wander and often trip validation / egress / retry in the guards column.
QUESTION = "show me the onboarding checklist"


def _guardrail_flags(t: tracing.Trace) -> str:
    """Which harness layers visibly fired on this run."""
    flags: list[str] = []
    for s in t.spans:
        if s.kind == "guardrail" or s.name == "guardrail":
            rule = s.attributes.get("rule", "")
            if "budget" in str(rule) or (s.error and "budget" in (s.error or "")):
                flags.append("budget")
            elif "precondition" in str(rule) or (
                s.error and "precondition" in (s.error or "")
            ):
                flags.append("precondition")
            elif "permission" in str(rule):
                flags.append("permission")
            elif "egress" in str(rule) or "secret" in str(rule):
                flags.append("egress")
            else:
                flags.append(str(rule) or "guardrail")
        if s.kind == "execute_tool" and s.error:
            err = s.error or ""
            if "no such tool" in err or "missing required" in err or "unexpected" in err:
                flags.append("validation")
            retries = s.attributes.get("gen_ai.tool.retries", 0)
            if retries:
                flags.append("retry")
        if s.attributes.get("gen_ai.tool.retries"):
            flags.append("retry")
    if t.stopped_because.startswith("budget:"):
        flags.append("budget")
    # de-dupe, stable order
    seen: list[str] = []
    for f in flags:
        if f and f not in seen:
            seen.append(f)
    return ",".join(seen) if seen else "-"


def _tokens(t: tracing.Trace) -> str:
    prompt = completion = 0
    for s in t.spans:
        prompt += int(s.attributes.get("gen_ai.usage.prompt_tokens") or 0)
        completion += int(s.attributes.get("gen_ai.usage.completion_tokens") or 0)
    if prompt or completion:
        return f"{prompt}+{completion}"
    return "-"


def _row(label: str, model_name: str, result) -> dict[str, str]:
    t = result.trace
    answer = (result.output or "").replace("\n", " ")
    if len(answer) > 80:
        answer = answer[:77] + "..."
    seq = " -> ".join(t.tool_sequence) if t.tool_sequence else "(none)"
    return {
        "run": label,
        "model": model_name,
        "steps": str(t.steps),
        "tools": seq,
        "tokens": _tokens(t),
        "cost": f"${t.cost_usd:.5f}",
        "answer": answer,
        "guards": _guardrail_flags(t),
    }


def _print_table(rows: list[dict[str, str]]) -> None:
    cols = ["run", "model", "steps", "tools", "tokens", "cost", "guards", "answer"]
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    # Cap wide columns so a terminal stays readable
    widths["tools"] = min(widths["tools"], 48)
    widths["answer"] = min(widths["answer"], 80)
    widths["model"] = min(widths["model"], 28)

    def cell(c: str, v: str) -> str:
        w = widths[c]
        if len(v) > w:
            v = v[: max(0, w - 3)] + "..."
        return v.ljust(w)

    header = "  ".join(cell(c, c) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(cell(c, r[c]) for c in cols))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare mock vs live paths on one fixed question.")
    parser.add_argument(
        "-n", type=int, default=3,
        help="number of live OpenRouter runs (default 3)")
    parser.add_argument(
        "--question", default=QUESTION,
        help="override the fixed question")
    args = parser.parse_args(argv)

    question = args.question
    print(f"question: {question}\n")

    rows: list[dict[str, str]] = []

    # ---- mock once -------------------------------------------------------
    db.reset()
    mock = MockModel(seed=0)
    mock_result = Agent(model=mock, config=AgentConfig()).run(
        question, case_id="variance-mock")
    tracing.write(mock_result.trace)
    rows.append(_row("mock-0", mock.name, mock_result))

    # ---- live N times ----------------------------------------------------
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        _print_table(rows)
        print()
        print("Live rows skipped: put OPENROUTER_API_KEY in .env at the repo root to run OpenRouterModel.")
        print("Mock row above is free and deterministic; live rows need a key.")
        return 0

    for i in range(max(0, args.n)):
        db.reset()
        live = OpenRouterModel()
        cfg = AgentConfig(
            budget=guards.Budget(max_usd_per_run=0.05),
        )
        try:
            result = Agent(model=live, config=cfg).run(
                question, case_id=f"variance-live-{i}")
        except Exception as e:  # noqa: BLE001 — show the row, don't crash the table
            print(f"live-{i} failed: {e}", file=sys.stderr)
            continue
        tracing.write(result.trace)
        rows.append(_row(f"live-{i}", live.name, result))

    _print_table(rows)
    print()
    print("Same input, different paths. Traces written to traces/runs.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
