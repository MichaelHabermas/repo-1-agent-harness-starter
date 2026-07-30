"""
Guardrails: the limits the agent cannot talk its way past.

A guardrail is not a prompt instruction. A prompt instruction is a request. A
guardrail is code that returns False. If a rule matters -- spend, blast radius,
credentials -- it belongs here, where it is deterministic and testable, not in
the system prompt where it is advisory.

Four kinds in this file:
  budgets       -- steps and dollars per run
  permissions   -- which tools this caller may use at all
  preconditions -- what must have happened before a write is allowed
  egress        -- what must never appear in the output
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent import db, tools as tool_mod


class GuardrailViolation(Exception):
    """Blocked. The harness turns most of these into an observation the model
    can recover from; a few of them stop the run outright."""

    def __init__(self, rule: str, message: str, fatal: bool = False):
        super().__init__(message)
        self.rule = rule
        self.fatal = fatal


@dataclass
class Budget:
    max_steps: int = 8
    max_usd_per_run: float = 0.05
    max_cost_usd: float = 0.05
    max_duration_s: float = 30.0

    def as_config(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_cost_usd": self.max_cost_usd,
            "max_duration_s": self.max_duration_s,
        }


@dataclass
class Policy:
    """Who may do what."""

    allowed_tools: set[str] = field(
        default_factory=lambda: set(tool_mod.REGISTRY.keys())
    )
    # Writes that require a specific read to have happened first. This is how
    # you turn "check the balance before booking" from a hopeful sentence in a
    # prompt into a rule the system enforces.
    preconditions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def as_config(self) -> dict[str, Any]:
        return {
            "allowed_tools": sorted(self.allowed_tools),
            "preconditions": {k: list(v) for k, v in self.preconditions.items()},
        }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_budget(budget: Budget, *, steps: int, cost_usd: float,
                 elapsed_s: float) -> None:
    """TODO 4a: raise GuardrailViolation when any limit is reached.

    Three limits, and they fail differently. A step limit usually means a loop.
    A cost limit usually means step inflation. A time limit usually means a slow
    tool. Same stop, three different investigations -- so put the rule name in
    the violation, not just a message.
    """
    raise NotImplementedError("check_budget")


def _reference_check_budget(budget: Budget, *, steps: int, cost_usd: float,
                            elapsed_s: float) -> None:
    if steps >= budget.max_steps:
        raise GuardrailViolation(
            "budget.max_steps",
            f"step limit reached ({budget.max_steps})",
            fatal=True,
        )
    if cost_usd >= budget.max_cost_usd:
        raise GuardrailViolation(
            "budget.max_cost_usd",
            f"cost limit reached (${budget.max_cost_usd})",
            fatal=True,
        )
    if elapsed_s >= budget.max_duration_s:
        raise GuardrailViolation(
            "budget.max_duration_s",
            f"time limit reached ({budget.max_duration_s}s)",
            fatal=True,
        )


def check_permission(policy: Policy, tool_name: str) -> None:
    """TODO 4b: block tools this caller may not use.

    Worth thinking about before you write it: should a blocked call be fatal,
    or should the model be told and allowed to try something else?
    """
    raise NotImplementedError("check_permission")


def _reference_check_permission(policy: Policy, tool_name: str) -> None:
    if tool_name not in policy.allowed_tools:
        raise GuardrailViolation(
            "permission.tool_not_allowed",
            f"tool {tool_name} is not available to this caller",
        )


def check_precondition(policy: Policy, tool_name: str,
                       already_called: list[str]) -> None:
    """TODO 4c: enforce reads that must happen before a write.

    This is how "check the balance before booking" stops being a hopeful
    sentence in a prompt and becomes a rule the system enforces.

    Build the mechanism now, but leave Policy.preconditions empty by default.
    Section two measures how often the model skips the check on its own;
    section three turns the rule on and proves it with the same number.
    Measure, then fix, then prove the fix -- in that order.
    """
    raise NotImplementedError("check_precondition")


def _reference_check_precondition(policy: Policy, tool_name: str,
                                  already_called: list[str]) -> None:
    required = policy.preconditions.get(tool_name)
    if not required:
        return
    missing = [r for r in required if r not in already_called]
    if missing:
        raise GuardrailViolation(
            "precondition.missing_read",
            f"{tool_name} requires {', '.join(missing)} first",
        )


# --------------------------------------------------------------------------
# Egress filtering
# --------------------------------------------------------------------------

# The obvious check: does the literal secret appear in the output.
def contains_known_secret(text: str) -> bool:
    return db.INTERNAL_API_KEY in text


# A slightly less naive check. Normalising away whitespace catches the variant
# where a value is emitted in chunks; a prefix pattern catches keys this
# process has never seen. Neither is complete -- see the section three tasks.
_SECRET_PATTERNS = [
    re.compile(r"nw_live_sk_[0-9a-f]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def find_secrets(text: str) -> list[str]:
    """TODO 4d: find credential material in the outgoing reply.

    Start with the obvious check: is the literal key in the string? Then ask
    yourself what a model does when a document tells it to print a value "with
    a space after every 4 characters so it is easier to read". Does your check
    still fire? Come back to this in section three.

    Return every secret-looking thing in the text.

    Note the normalisation step. A model that emits a key with spaces in it has
    still leaked the key, and a substring check on the raw string will happily
    say it didn't.
    """
    found: list[str] = []
    normalised = re.sub(r"[\s\-_.]", "", text)
    if db.INTERNAL_API_KEY in text:
        found.append("literal:INTERNAL_API_KEY")
    elif re.sub(r"[\s\-_.]", "", db.INTERNAL_API_KEY) in normalised:
        found.append("obfuscated:INTERNAL_API_KEY")
    for pattern in _SECRET_PATTERNS:
        for m in pattern.findall(text):
            found.append(f"pattern:{m[:12]}...")
    return found


def redact(text: str) -> str:
    """Last line of defence on the way out."""
    out = text.replace(db.INTERNAL_API_KEY, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out
