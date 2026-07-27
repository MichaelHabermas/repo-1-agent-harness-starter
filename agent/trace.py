"""
Tracing.

A trace is the complete record of one run -- not a log line. If it isn't in
here, you cannot evaluate it, you cannot reproduce it, you cannot cost it and
you cannot show it to anyone who asks.

Attribute names follow the OpenTelemetry GenAI semantic conventions (the
`gen_ai.*` namespace) so this data has the same shape as what a real backend
expects. Those conventions are still pre-stable, so we pin the version we
built against rather than tracking the tip.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEMCONV_VERSION = "1.42.0"      # pin it; the conventions still move


@dataclass
class Span:
    name: str                              # gen_ai.operation.name
    kind: str                              # "chat" | "execute_tool" | "invoke_agent"
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return round((end - self.started_at) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "attributes": self.attributes,
        }


@dataclass
class Trace:
    """One run, start to finish."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    case_id: str | None = None
    attempt: int = 0
    input: str = ""
    output: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    stopped_because: str = "completed"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    # -- recording -------------------------------------------------------

    def span(self, name: str, kind: str, **attributes: Any) -> Span:
        s = Span(name=name, kind=kind, started_at=time.time(), attributes=dict(attributes))
        self.spans.append(s)
        return s

    def finish(self, span: Span, error: str | None = None, **attributes: Any) -> None:
        span.ended_at = time.time()
        span.error = error
        span.attributes.update(attributes)

    # -- reading (this is what graders use) ------------------------------

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Every tool call, in order. The trajectory."""
        return [
            {
                "tool": s.attributes.get("gen_ai.tool.name"),
                "args": s.attributes.get("gen_ai.tool.arguments", {}),
                "ok": s.error is None,
                "error": s.error,
                "duration_ms": s.duration_ms,
            }
            for s in self.spans
            if s.kind == "execute_tool"
        ]

    @property
    def tool_sequence(self) -> list[str]:
        return [c["tool"] for c in self.tool_calls]

    @property
    def steps(self) -> int:
        return len(self.tool_calls)

    @property
    def cost_usd(self) -> float:
        return round(sum(s.attributes.get("cost.usd", 0.0) for s in self.spans), 6)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return round((end - self.started_at) * 1000, 2)

    @property
    def retrieved_docs(self) -> list[str]:
        """Document ids the run actually saw. Needed to prove where an answer
        came from -- and to prove where an injection came from."""
        ids: list[str] = []
        for s in self.spans:
            for doc_id in s.attributes.get("retrieved.doc_ids", []) or []:
                ids.append(doc_id)
        return ids

    def state_changed(self) -> bool:
        return self.state_before != self.state_after

    # -- serialising -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "attempt": self.attempt,
            "semconv_version": SEMCONV_VERSION,
            "input": self.input,
            "output": self.output,
            "config": self.config,
            "stopped_because": self.stopped_because,
            "steps": self.steps,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "tool_sequence": self.tool_sequence,
            "tool_calls": self.tool_calls,
            "retrieved_doc_ids": self.retrieved_docs,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "spans": [s.to_dict() for s in self.spans],
        }

    def summary(self) -> str:
        arrow = " -> ".join(self.tool_sequence) or "(no tools)"
        return (
            f"{self.trace_id}  steps={self.steps}  ${self.cost_usd:.4f}  "
            f"{self.duration_ms:.0f}ms  [{arrow}]"
        )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

DEFAULT_PATH = Path("traces/runs.jsonl")


def write(trace: Trace, path: Path | str = DEFAULT_PATH) -> None:
    """Append one trace as a line of JSON. Boring on purpose: JSONL is
    grep-able, diff-able, and loads into anything."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(trace.to_dict(), default=str) + "\n")


def read_all(path: Path | str = DEFAULT_PATH) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
