# Section 1 — Tasks

**Time: about 45 minutes.** Do them in order. Each one is independently
testable, and `tests/test_agent.py` already expects all five.

```bash
python tests/test_agent.py     # 6 of 9 pass right now
```

---

## 0. Read the trace (5 min, no code)

```bash
python -m app.cli --trace "book time off from 2026-09-14 to 2026-09-18, 5 days"
```

Write down three things the trace should tell you and doesn't. Keep the list;
you'll check it at the end.

---

## 1. Context — keep the window bounded

`agent/harness.py`, TODO 1.

Call `ctx.compact()` in the loop and record it when turns get dropped. Look at
`ContextPolicy` in `agent/context.py` first — the policy is four numbers you can
version, which is the point.

**Ask yourself:** compaction drops from the middle and keeps the first user
turn. Why the middle? What would break if you dropped from the end instead?

---

## 2. Tools — validate before dispatch

`agent/harness.py`, TODO 2 in `_dispatch`.

Right now `python -m app.cli "who is nobody"` raises straight through to the
user. Call `tool_mod.validate_call()` first, and turn `ToolError` into an
observation the model can read.

**The principle:** an exception that reaches the user is a harness bug. An error
the model can read is information it can recover from. Errors are data.

---

## 3. Verification — retry reads, never retry writes

`agent/harness.py`, TODO 3.

Retry on `spec.effect == "read"`. Do not retry writes.

**Before you code it:** why the asymmetry? What happens if
`submit_pto_request` succeeds and the *response* is what fails?

---

## 4. Guardrails — budgets, permissions, egress

`agent/guardrails.py`, TODOs 4a–4d, then wire them into the loop.

- `check_budget` — steps, dollars, wall clock. Put the rule name in the
  violation; a step limit and a cost limit lead to different investigations.
- `check_permission` — tools this caller may not use.
- `check_precondition` — reads that must happen before a write.
- `find_secrets` — start with the literal check.
- Then wire the egress check into `run()`. What should the user see when you
  stop early? Silence is not an answer.

Now try:

```bash
python -m app.cli "show me the onboarding checklist"
```

The handbook contains a document that gives the assistant instructions. Your
egress check should catch what happens next.

**Build `check_precondition`, but leave `Policy.preconditions` empty by
default.** Section two measures how often the model skips the balance check on
its own; section three turns the rule on and proves it with the same number.
Measure, then fix, then prove the fix — in that order.

---

## 5. Trace — record all of it

`agent/harness.py`, TODO 5 (three places) and `AgentConfig.as_dict`.

Capture `db.STATE.snapshot()` before and after. Open a span for the model call
and one per tool call. Put everything that changes behaviour into the config.

**The test:** could a colleague reproduce this exact run from the trace alone?
If a number in the config is missing, they can't — and neither can you, in three
weeks, when the score moves and nobody remembers changing anything.

---

## Done when

```bash
python tests/test_agent.py                    # 9 passed
python -m app.cli --trace "what is the expense policy?"
```

The trace now shows three handbook searches for one question. You can see it.
You still can't prove it's a problem, put a number on it, or stop someone
shipping it.

That's section two.
