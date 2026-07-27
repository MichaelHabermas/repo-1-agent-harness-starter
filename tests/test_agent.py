"""
Unit tests for the harness. Not evals -- these are the ordinary tests that keep
the machinery honest so your eval results mean something.

    python -m pytest tests/ -q       (if you have pytest)
    python tests/test_agent.py       (if you don't)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import context as ctx
from agent import db, guardrails as guards, tools as tool_mod
from agent.harness import Agent, AgentConfig
from agent.model import MockModel, Turn


def test_reset_restores_state():
    db.reset()
    before = db.STATE.snapshot()
    tool_mod.submit_pto_request("e-1001", "2026-09-01", "2026-09-02", 2)
    assert db.STATE.snapshot() != before
    db.reset()
    assert db.STATE.snapshot() == before


def test_validate_call_rejects_unknown_tool():
    try:
        tool_mod.validate_call("delete_everything", {})
    except tool_mod.ToolError as e:
        assert "no such tool" in str(e)
    else:
        raise AssertionError("expected ToolError")


def test_validate_call_rejects_missing_argument():
    try:
        tool_mod.validate_call("get_pto_balance", {})
    except tool_mod.ToolError as e:
        assert "missing required" in str(e)
    else:
        raise AssertionError("expected ToolError")


def test_budget_stops_a_runaway_loop():
    db.reset()
    cfg = AgentConfig(budget=guards.Budget(max_steps=2))
    result = Agent(model=MockModel(seed=0), config=cfg).run(
        "what is the expense policy?")
    assert result.trace.steps <= 2
    assert result.trace.stopped_because.startswith("budget:")


def test_precondition_blocks_write_without_read():
    db.reset()
    policy = guards.Policy(preconditions={"submit_pto_request": ("get_pto_balance",)})
    cfg = AgentConfig(policy=policy)
    # seed 1 is one of the seeds where the model skips the balance check
    result = Agent(model=MockModel(seed=1), config=cfg).run(
        "book time off from 2026-09-14 to 2026-09-18, 5 days")
    assert len(db.STATE.pto_requests) <= 1
    blocked = [s for s in result.trace.spans
               if s.error and "precondition" in (s.error or "")]
    assert blocked, "expected the precondition guardrail to fire"


def test_read_tools_are_retried_writes_are_not():
    read = tool_mod.REGISTRY["get_pto_balance"]
    write = tool_mod.REGISTRY["submit_pto_request"]
    assert read.effect == "read"
    assert write.effect == "write"


def test_context_compaction_keeps_the_original_request():
    policy = ctx.ContextPolicy(compact_after=4, max_turns=4)
    turns = [Turn(role="user", content="the original question")]
    turns += [Turn(role="observation", content=f"obs {i}", tool="t") for i in range(10)]
    out, dropped = ctx.compact(turns, policy)
    assert dropped > 0
    assert out[0].content == "the original question"
    assert any("compacted" in t.content for t in out)


def test_trace_records_the_trajectory_and_state():
    db.reset()
    result = Agent(model=MockModel(seed=0)).run("how many days off do I have?")
    d = result.trace.to_dict()
    assert d["tool_sequence"] == ["get_pto_balance"]
    assert d["state_before"] == d["state_after"]
    assert d["config"]["prompt_version"]
    assert d["cost_usd"] > 0


def test_secret_detector_sees_through_spacing():
    key = db.INTERNAL_API_KEY
    spaced = " ".join(key[i:i + 4] for i in range(0, len(key), 4))
    assert guards.find_secrets(key)
    assert guards.find_secrets(spaced), "a spaced key is still a leaked key"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n  {passed} passed, {failed} failed\n")
    raise SystemExit(1 if failed else 0)
