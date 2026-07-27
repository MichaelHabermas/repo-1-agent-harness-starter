"""
The harness -- STARTER.

Agent = model + harness. The model is the stateless part that decides. The
harness is everything else, and it is all yours.

What's here works. Ask it "how many days off do I have?" and it answers. That
is the trap: it works on the happy path, and the happy path is not the job.

Your task is the five layers. Each TODO below is one of them. Do them in order;
each one is testable on its own, and tests/test_agent.py already expects them.

  TODO 1  context      compact the window as a run grows          (agent/context.py)
  TODO 2  tools        validate before dispatch, errors as data   (agent/tools.py)
  TODO 3  verification retry reads, never retry writes
  TODO 4  guardrails   budgets, permissions, egress               (agent/guardrails.py)
  TODO 5  trace        record all of it                           (agent/trace.py)

Run `python -m app.cli --trace "what is the expense policy?"` before you start
and after each TODO. Watch what appears in the trace. When you can answer
"what exactly did it do, and what did it cost?" from the trace alone, you're done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent import context as ctx
from agent import db, guardrails as guards
from agent import tools as tool_mod
from agent import trace as tracing
from agent.model import Final, Model, MockModel, ToolCall, Turn


@dataclass
class AgentConfig:
    """Everything that changes behaviour, in one versioned object.

    If it isn't in here it can't go in the trace, and if it isn't in the trace
    you can't reproduce the run. "We changed the prompt" is the most common
    unlogged cause of a score moving.
    """

    prompt_version: str = "system-v3"
    budget: guards.Budget = field(default_factory=guards.Budget)
    policy: guards.Policy = field(default_factory=guards.Policy)
    context_policy: ctx.ContextPolicy = field(default_factory=ctx.ContextPolicy)
    redact_output: bool = True
    max_tool_retries: int = 1

    def as_dict(self, model_name: str) -> dict[str, Any]:
        # TODO 5: this is what lands in the trace. Is everything that changes
        # behaviour represented here? What's missing?
        return {"model": model_name, "prompt_version": self.prompt_version}


@dataclass
class Result:
    output: str
    trace: tracing.Trace

    @property
    def ok(self) -> bool:
        return self.trace.stopped_because == "completed"


class Agent:
    def __init__(self, model: Model | None = None,
                 config: AgentConfig | None = None,
                 actor_id: str = "e-1001"):
        self.model = model or MockModel()
        self.config = config or AgentConfig()
        self.actor_id = actor_id

    def run(self, user_message: str, *, case_id: str | None = None,
            attempt: int = 0) -> Result:
        actor = db.STATE.employees.get(self.actor_id, {"id": self.actor_id})
        system = ctx.build_system(actor)

        t = tracing.Trace(
            case_id=case_id,
            attempt=attempt,
            input=user_message,
            config=self.config.as_dict(getattr(self.model, "name", "unknown")),
        )
        # TODO 5: an eval cannot assert on a state change you never recorded.
        #         Capture db.STATE.snapshot() here and again at the end.

        turns: list[Turn] = [Turn(role="user", content=user_message)]
        answer = ""

        # A loop with a fixed bound and no accounting. It cannot run forever,
        # which is something, and it cannot tell you why it stopped, which is
        # the problem.
        for _ in range(10):

            # TODO 4: before spending another step, can you afford it?
            #         guards.check_budget(...) -- steps, dollars, wall clock.
            #         What should the user see when you stop early? Silence is
            #         not an answer.

            # TODO 1: turns, dropped = ctx.compact(turns, self.config.context_policy)
            #         Record it when it happens. A silent truncation is a bug
            #         you will never find.

            decision = self.model.decide(system, turns)

            # TODO 5: record the model call as a span. Which model, how many
            #         turns went in, what came back, what it cost.

            if isinstance(decision, Final):
                answer = decision.text
                break

            turns.append(self._dispatch(t, decision))

        # TODO 4: egress. The model can be talked into repeating things it was
        #         given. guards.find_secrets(answer) tells you if that happened.
        #         What do you do about it, and where do you record that you did?

        t.output = answer
        t.ended_at = time.time()
        return Result(output=answer, trace=t)

    def _dispatch(self, t: tracing.Trace, call: ToolCall) -> Turn:
        """One tool call.

        Right now an unknown tool name or a missing argument raises straight
        through to the caller. Try: `python -m app.cli "who is nobody"`.

        An exception that reaches the user is a harness bug. An error the model
        can read is information it can recover from.
        """
        # TODO 5: open a span here with the tool name and its arguments.
        # TODO 4: guards.check_permission / guards.check_precondition
        # TODO 2: tool_mod.validate_call(call.name, call.args) before you run it,
        #         and catch tool_mod.ToolError into an observation Turn
        # TODO 3: retry reads (spec.effect == "read"), never retry writes.
        #         Ask yourself why that asymmetry matters before you code it.

        spec = tool_mod.REGISTRY[call.name]
        data = spec.fn(**call.args)
        return Turn(role="observation", content=_render(data), tool=call.name,
                    ok=True, data=data if isinstance(data, dict) else None)


def _render(data: Any) -> str:
    if isinstance(data, dict) and "results" in data:
        docs = data["results"]
        if not docs:
            return "no matching documents"
        return "\n".join(f"[{d['id']}] {d['title']}: {d['body']}" for d in docs)
    return str(data)
