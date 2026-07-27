"""
The tools the agent can call.

Each tool declares a schema. The harness uses the schema for two things:
dispatch (does this call even typecheck?) and permissions (is this call allowed
in this context?). Schemas are also what you contract-test -- a tool whose shape
changes without warning is an external dependency that broke your agent.

Tools are grouped by effect:
  READ  -- safe, idempotent, cheap to retry
  WRITE -- changes company state, must never be retried blindly
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from agent import db

Effect = Literal["read", "write"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    effect: Effect
    description: str
    params: dict[str, str]          # name -> "type: what it is"
    required: tuple[str, ...]
    fn: Callable[..., Any]
    cost_usd: float = 0.0002        # pretend per-call cost, so budgets bite


class ToolError(Exception):
    """Raised for a bad call. The harness turns this into an observation the
    model can read and recover from -- errors are information, not crashes."""


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------


def search_handbook(query: str) -> dict[str, Any]:
    """Keyword search over the company handbook."""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    terms = {t for t in query.lower().split() if len(t) > 2}
    hits = []
    for doc in db.STATE.handbook:
        haystack = f"{doc['title']} {doc['tags']}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            hits.append((score, doc))
    hits.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return {
        "query": query,
        "results": [
            {"id": d["id"], "title": d["title"], "body": d["body"]}
            for _, d in hits[:3]
        ],
    }


def lookup_employee(email: str) -> dict[str, Any]:
    """Find someone in the company directory by email address."""
    if not isinstance(email, str) or "@" not in email:
        raise ToolError("email must look like an email address")
    for emp in db.STATE.employees.values():
        if emp["email"].lower() == email.lower():
            return dict(emp)
    raise ToolError(f"no employee found with email {email}")


def get_pto_balance(employee_id: str) -> dict[str, Any]:
    """How many paid days off this person has left."""
    rec = db.STATE.pto.get(employee_id)
    if rec is None:
        raise ToolError(f"unknown employee_id {employee_id}")
    remaining = rec["accrued_days"] - rec["used_days"] - rec["pending_days"]
    return {
        "employee_id": employee_id,
        "accrued_days": rec["accrued_days"],
        "used_days": rec["used_days"],
        "pending_days": rec["pending_days"],
        "remaining_days": round(remaining, 1),
    }


def submit_pto_request(employee_id: str, start_date: str, end_date: str,
                       days: float) -> dict[str, Any]:
    """Book time off. THIS CHANGES COMPANY STATE."""
    rec = db.STATE.pto.get(employee_id)
    if rec is None:
        raise ToolError(f"unknown employee_id {employee_id}")
    try:
        days = float(days)
    except (TypeError, ValueError):
        raise ToolError("days must be a number")
    if days <= 0:
        raise ToolError("days must be positive")

    # Note what this tool does NOT do: it does not refuse an overdraw. The
    # business rule ("you cannot book more days than you have remaining") lives
    # in the handbook, not in the API. That gap is deliberate. It is exactly the
    # kind of rule that has to be enforced by the harness or caught by an eval,
    # because the tool will happily let the agent break it.
    request = {
        "request_id": f"pto-{len(db.STATE.pto_requests) + 1:04d}",
        "employee_id": employee_id,
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "status": "pending_manager_approval",
        "submitted_at": time.strftime("%Y-%m-%d"),
    }
    db.STATE.pto_requests.append(request)
    rec["pending_days"] += days
    return request


def escalate_to_human(reason: str, summary: str = "") -> dict[str, Any]:
    """Hand the conversation to a person. Always a valid, safe answer."""
    if not isinstance(reason, str) or not reason.strip():
        raise ToolError("reason must be a non-empty string")
    ticket = {
        "ticket_id": f"esc-{len(db.STATE.escalations) + 1:04d}",
        "reason": reason,
        "summary": summary,
        "queue": "people-ops",
    }
    db.STATE.escalations.append(ticket)
    return ticket


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="search_handbook",
            effect="read",
            description="Search the company handbook for a policy.",
            params={"query": "string: what to look for"},
            required=("query",),
            fn=search_handbook,
        ),
        ToolSpec(
            name="lookup_employee",
            effect="read",
            description="Look up an employee in the directory by email.",
            params={"email": "string: work email address"},
            required=("email",),
            fn=lookup_employee,
        ),
        ToolSpec(
            name="get_pto_balance",
            effect="read",
            description="Get remaining paid time off for an employee.",
            params={"employee_id": "string: e.g. e-1001"},
            required=("employee_id",),
            fn=get_pto_balance,
        ),
        ToolSpec(
            name="submit_pto_request",
            effect="write",
            description="Submit a time-off request for manager approval.",
            params={
                "employee_id": "string: e.g. e-1001",
                "start_date": "string: YYYY-MM-DD",
                "end_date": "string: YYYY-MM-DD",
                "days": "number: working days requested",
            },
            required=("employee_id", "start_date", "end_date", "days"),
            fn=submit_pto_request,
            cost_usd=0.001,
        ),
        ToolSpec(
            name="escalate_to_human",
            effect="write",
            description="Hand off to a human when you are not confident.",
            params={
                "reason": "string: why you are escalating",
                "summary": "string: what the user asked for",
            },
            required=("reason",),
            fn=escalate_to_human,
        ),
    ]
}


def describe_tools() -> str:
    """The tool documentation the model sees. Real harnesses send JSON schema;
    the shape of the problem is the same."""
    lines = []
    for spec in REGISTRY.values():
        args = ", ".join(f"{k} ({v})" for k, v in spec.params.items())
        lines.append(f"- {spec.name}({args}) [{spec.effect}] -- {spec.description}")
    return "\n".join(lines)


def validate_call(name: str, args: dict[str, Any]) -> None:
    """Cheap schema check before dispatch. Catches the model inventing a tool or
    forgetting an argument, which is a surprisingly large share of real failures."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ToolError(f"no such tool: {name}")
    missing = [p for p in spec.required if p not in args]
    if missing:
        raise ToolError(f"{name} missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in spec.params]
    if unknown:
        raise ToolError(f"{name} got unexpected argument(s): {', '.join(unknown)}")
