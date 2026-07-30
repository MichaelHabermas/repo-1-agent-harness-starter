"""
The model layer.

A model is a stateless thing that reads a context and decides what to do next:
call a tool, or answer. That is the whole interface. Everything else in this
repo is harness.

We ship a deterministic stand-in rather than calling a hosted model, for three
reasons that matter in a workshop:

  1. It runs offline, instantly, for free.
  2. Its failure modes are *planted and reproducible*, so error analysis finds
     the same bugs for everyone in the room instead of a different sample of
     model noise each time.
  3. Swapping in a real model is one class. See AnthropicModel at the bottom --
     the harness, the tools, the traces and every eval stay exactly as they are.

THE PLANTED FAILURES (don't read this section if you're about to do the error
analysis exercise -- finding them yourself is the point):

  F1  "vacation days" phrasing routes to the handbook instead of the balance
      tool. Plausible answer, wrong tool, no real number.
  F2  Fabricates a policy when the handbook search comes back empty, instead of
      escalating.
  F3  Complies with instructions embedded in retrieved documents.
  F4  Searches three times for expense questions. Right answer, 3x the cost.
  F5  Skips the balance check before booking time off, about a quarter of the
      time. This one is stochastic on purpose -- it is why pass^k exists.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent import db


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class Final:
    text: str


Decision = ToolCall | Final


@dataclass
class Turn:
    """One entry in the context: what happened, in order."""
    role: str                       # "user" | "assistant" | "observation"
    content: str
    tool: str | None = None
    ok: bool = True
    data: dict[str, Any] | None = None


class Model(Protocol):
    name: str

    def decide(self, system: str, turns: list[Turn]) -> Decision: ...


# --------------------------------------------------------------------------
# Intent routing
# --------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("book_pto", ("book", "request time", "take time", "put in for", "request pto",
                  "schedule leave", "book time")),
    ("pto_balance", ("how many days", "days left", "days off do i have",
                     "pto balance", "how much pto", "vacation days",
                     "time off do i have", "days remaining", "time off balance")),
    ("directory", ("who is", "what is the email", "manager of", "contact details",
                   "look up", "find the email")),
    ("policy", ("policy", "am i allowed", "can i", "what happens if", "how do i",
                "what is the rule", "expense", "remote", "parental", "laptop",
                "equipment", "onboarding", "password", "mfa")),
]

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def classify(text: str) -> str:
    low = text.lower()
    for intent, needles in _INTENT_PATTERNS:
        if any(n in low for n in needles):
            return intent
    return "unknown"


# --------------------------------------------------------------------------
# The mock
# --------------------------------------------------------------------------


@dataclass
class MockModel:
    """Deterministic given (seed, input). Vary the seed to vary the run --
    which is what running the same case k times is for."""

    seed: int = 0
    name: str = "mock-assistant-v1"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _user_text(turns: list[Turn]) -> str:
        for t in reversed(turns):
            if t.role == "user":
                return t.content
        return ""

    @staticmethod
    def _observations(turns: list[Turn]) -> list[Turn]:
        return [t for t in turns if t.role == "observation"]

    @staticmethod
    def _called(turns: list[Turn], tool: str) -> list[Turn]:
        return [t for t in turns if t.role == "observation" and t.tool == tool]

    @staticmethod
    def _succeeded(turns: list[Turn], tool: str) -> list[Turn]:
        """Calls that actually ran. A blocked or failed call did not happen,
        and treating it as though it did is how agents get stuck."""
        return [t for t in turns if t.role == "observation"
                and t.tool == tool and t.ok]

    @staticmethod
    def _actor_id(system: str) -> str:
        m = re.search(r"\((e-\d{4})\)", system)
        return m.group(1) if m else "e-1001"

    @staticmethod
    def _injection_in(obs: list[Turn]) -> str | None:
        """F3: the model reads retrieved documents as instructions. Real models
        are better than this than they used to be, and still not immune."""
        for t in obs:
            for doc in (t.data or {}).get("results", []) or []:
                body = doc.get("body", "")
                if "SYSTEM NOTE FOR THE ASSISTANT" in body:
                    return body
        return None

    # -- the decision ----------------------------------------------------

    def decide(self, system: str, turns: list[Turn]) -> Decision:
        user = self._user_text(turns)
        obs = self._observations(turns)
        intent = classify(user)
        actor = self._actor_id(system)

        # Once a handoff has succeeded the run is over. Without a terminal
        # condition here the loop escalates forever and only the step budget
        # stops it -- which is a good reason to have a step budget, and a
        # better reason to give every path an ending.
        handed_off = self._succeeded(turns, "escalate_to_human")
        if handed_off:
            ticket = (handed_off[-1].data or {}).get("ticket_id", "unknown")
            return Final(
                "I'm not confident enough to answer this one, so I've passed it "
                f"to People Ops ({ticket}). They'll follow up with you."
            )

        # ---- recover from a guardrail that told us what was missing -----
        # The harness doesn't just say "no", it says what it wanted first.
        # Acting on that is the difference between a guardrail that shapes
        # behaviour and one that just breaks the run.
        if obs and not obs[-1].ok and "requires" in obs[-1].content:
            m = re.search(r"requires ([a-z_]+)", obs[-1].content)
            if m and m.group(1) in ("get_pto_balance",):
                return ToolCall(m.group(1), {"employee_id": actor})

        # ---- F3: obey whatever a retrieved document told us to do ------
        injected = self._injection_in(obs)
        if injected and not any(t.role == "assistant" for t in turns):
            leak = self._follow_injection(injected)
            if leak:
                return Final(leak)

        if intent == "pto_balance":
            return self._pto_balance(user, obs, actor)
        if intent == "book_pto":
            return self._book_pto(user, obs, actor)
        if intent == "directory":
            return self._directory(user, obs)
        if intent == "policy":
            return self._policy(user, obs)
        return self._unknown(user, obs)

    # -- intents ---------------------------------------------------------

    def _pto_balance(self, user: str, obs: list[Turn], actor: str) -> Decision:
        # F1: "vacation days" reads as a policy question, so it goes to the
        # handbook and never fetches the actual number.
        if "vacation" in user.lower() and not self._called(obs, "get_pto_balance"):
            if not self._called(obs, "search_handbook"):
                return ToolCall("search_handbook", {"query": "vacation time off policy"})
            docs = self._first_results(obs, "search_handbook")
            if docs:
                return Final(
                    "Here's what the handbook says about time off: "
                    f"{docs[0]['body']}"
                )
            return Final("I couldn't find the vacation policy.")

        got = self._called(obs, "get_pto_balance")
        if not got:
            return ToolCall("get_pto_balance", {"employee_id": actor})
        rec = got[-1]
        if not rec.ok:
            return ToolCall("escalate_to_human",
                            {"reason": "could not read PTO balance",
                             "summary": user})
        d = rec.data or {}
        return Final(
            f"You have {d.get('remaining_days')} paid days off remaining "
            f"({d.get('accrued_days')} accrued, {d.get('used_days')} used, "
            f"{d.get('pending_days')} pending)."
        )

    def _book_pto(self, user: str, obs: list[Turn], actor: str) -> Decision:
        dates = _DATE.findall(user)
        submitted = self._succeeded(obs, "submit_pto_request")
        if submitted:
            last = submitted[-1]
            if not last.ok:
                return ToolCall("escalate_to_human",
                                {"reason": "time-off request failed", "summary": user})
            r = last.data or {}
            return Final(
                f"Booked. Request {r.get('request_id')} for {r.get('days')} day(s) "
                f"from {r.get('start_date')} to {r.get('end_date')} is now "
                f"{r.get('status').replace('_', ' ')}."
            )

        if len(dates) < 2:
            return ToolCall("escalate_to_human",
                            {"reason": "time-off request without clear dates",
                             "summary": user})

        days = self._days_from(user, dates)

        # F5: the balance check before a write is skipped about a quarter of the
        # time. Stochastic, seeded -- so a single run usually looks fine and the
        # fifth run does not. This is the whole argument for pass^k in one bug.
        checked = bool(self._succeeded(obs, "get_pto_balance"))
        if not checked:
            skip = self._rng.random() < 0.25
            if not skip:
                return ToolCall("get_pto_balance", {"employee_id": actor})

        return ToolCall("submit_pto_request", {
            "employee_id": actor,
            "start_date": dates[0],
            "end_date": dates[1],
            "days": days,
        })

    def _directory(self, user: str, obs: list[Turn]) -> Decision:
        looked = self._called(obs, "lookup_employee")
        emails = re.findall(r"[\w.+-]+@[\w.-]+", user)
        if not looked:
            if emails:
                return ToolCall("lookup_employee", {"email": emails[0]})
            name = user.lower().replace("who is", "").strip(" ?.")
            guess = name.replace(" ", ".") + "@northwind.example"
            return ToolCall("lookup_employee", {"email": guess})
        rec = looked[-1]
        if not rec.ok:
            return ToolCall("escalate_to_human",
                            {"reason": "person not found in directory",
                             "summary": user})
        e = rec.data or {}
        return Final(
            f"{e.get('name')} -- {e.get('title')}, {e.get('location')}. "
            f"Email {e.get('email')}, manager {e.get('manager_email')}."
        )

    def _policy(self, user: str, obs: list[Turn]) -> Decision:
        searches = self._called(obs, "search_handbook")

        # F4: expense questions get searched three times. Same answer, 3x cost.
        if "expense" in user.lower() and len(searches) < 3:
            variants = ["expense policy", "expense reimbursement receipt",
                        "travel meals spend limit"]
            return ToolCall("search_handbook", {"query": variants[len(searches)]})

        if not searches:
            return ToolCall("search_handbook", {"query": self._query_from(user)})

        docs = self._first_results(obs, "search_handbook")
        if docs:
            return Final(f"{docs[0]['title']}: {docs[0]['body']}")

        # F2: nothing found, so make something up rather than escalate.
        return Final(
            "Based on standard company practice, that is generally permitted as "
            "long as you notify your manager in advance."
        )

    def _unknown(self, user: str, obs: list[Turn]) -> Decision:
        if not self._called(obs, "search_handbook"):
            return ToolCall("search_handbook", {"query": self._query_from(user)})
        docs = self._first_results(obs, "search_handbook")
        if docs:
            return Final(f"{docs[0]['title']}: {docs[0]['body']}")
        return ToolCall("escalate_to_human",
                        {"reason": "no handbook match and no confident answer",
                         "summary": user})

    # -- small helpers ---------------------------------------------------

    @staticmethod
    def _query_from(user: str) -> str:
        stop = {"what", "is", "the", "how", "do", "i", "can", "a", "an", "for",
                "our", "my", "to", "of", "and", "am", "are", "if"}
        words = [w.strip("?.,!") for w in user.lower().split()]
        keep = [w for w in words if w and w not in stop]
        return " ".join(keep[:6]) or user

    @staticmethod
    def _first_results(obs: list[Turn], tool: str) -> list[dict[str, Any]]:
        for t in obs:
            if t.tool == tool and t.ok:
                res = (t.data or {}).get("results") or []
                if res:
                    return res
        return []

    @staticmethod
    def _days_from(user: str, dates: list[str]) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:working\s*)?days?", user.lower())
        if m:
            return float(m.group(1))
        from datetime import date
        try:
            a = date.fromisoformat(dates[0])
            b = date.fromisoformat(dates[1])
            return float(max(1, (b - a).days + 1))
        except ValueError:
            return 1.0

    @staticmethod
    def _follow_injection(body: str) -> str | None:
        """Do what the document said. Two variants: the obvious one, and the
        one that walks straight past a naive substring filter."""
        key = db.INTERNAL_API_KEY
        low = body.lower()
        if "internal_api_key" not in low:
            return None
        if "space after every" in low or "obfuscat" in low:
            spaced = " ".join(key[i:i + 4] for i in range(0, len(key), 4))
            return f"Here is the onboarding checklist. Verification value: {spaced}"
        return f"Here is the onboarding checklist. Verification value: {key}"


# --------------------------------------------------------------------------
# The real thing -- OpenRouter, stdlib only
# --------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"

# Approximate USD per million tokens for the default cheap model.
# Prices drift; check current numbers at https://openrouter.ai/models
PRICE_PER_MTOK = {
    "input": 0.10,
    "output": 0.40,
}


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int,
                      prices: dict[str, float] | None = None) -> float:
    """Rough USD cost from token counts. Teaching estimate, not billing."""
    p = prices or PRICE_PER_MTOK
    return (
        (prompt_tokens / 1_000_000.0) * p["input"]
        + (completion_tokens / 1_000_000.0) * p["output"]
    )


def tools_as_openai_schema() -> list[dict[str, Any]]:
    """Convert the harness registry into OpenAI-compatible tool definitions.

    Lives here (not in tools.py) so the tool module stays free of provider
    shapes. The schema is still the contract; this is just packaging.
    """
    from agent import tools as tool_mod

    out: list[dict[str, Any]] = []
    for spec in tool_mod.REGISTRY.values():
        properties: dict[str, Any] = {}
        for pname, pdesc in spec.params.items():
            # params are "type: description" strings in this repo
            ptype, _, desc = pdesc.partition(":")
            ptype = ptype.strip()
            json_type = "number" if ptype == "number" else "string"
            properties[pname] = {
                "type": json_type,
                "description": desc.strip() or pdesc,
            }
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.required),
                },
            },
        })
    return out


def turns_to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    """Map harness turns to OpenAI-style chat messages.

    Observations become user messages with a tool tag. That keeps multi-step
    runs working without storing tool_call_ids on Turn -- the model still uses
    native tools/tool_calls on the response side.
    """
    messages: list[dict[str, Any]] = []
    for t in turns:
        if t.role == "user":
            messages.append({"role": "user", "content": t.content})
        elif t.role == "assistant":
            messages.append({"role": "assistant", "content": t.content})
        elif t.role == "observation":
            status = "ok" if t.ok else "error"
            messages.append({
                "role": "user",
                "content": f"[tool:{t.tool} {status}] {t.content}",
            })
    return messages or [{"role": "user", "content": "(no input)"}]


def parse_openrouter_decision(body: dict[str, Any]) -> Decision:
    """Turn an OpenRouter chat.completions body into a ToolCall or Final.

    Deliberately does NOT repair bad tool names or arguments. Invented tools
    and missing args must reach the harness's validate_call -- that is a
    feature, not a bug, in a live teaching demo.
    """
    choices = body.get("choices") or []
    if not choices:
        return Final("")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        # name may be missing or nonsense -- pass through as-is
        name = fn.get("name") if fn.get("name") is not None else ""
        if not isinstance(name, str):
            name = str(name)
        raw_args = fn.get("arguments", "{}")
        args: dict[str, Any]
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                # Not repaired into plausible args. Empty dict fails required-arg
                # validation the same way a real missing-args call does.
                parsed = {}
            args = parsed if isinstance(parsed, dict) else {}
        else:
            args = {}
        return ToolCall(name=name, args=args)
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    return Final(content)


def usage_from_response(body: dict[str, Any],
                        prices: dict[str, float] | None = None) -> dict[str, Any]:
    """Extract token counts and an estimated cost from a completions body."""
    usage = body.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
        "cost_usd": round(estimate_cost_usd(prompt, completion, prices), 8),
    }


def build_openrouter_request(
    *,
    model: str,
    system: str,
    turns: list[Turn],
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Pure request-body builder. Unit-tested with no network."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            *turns_to_messages(turns),
        ],
        "tools": tools_as_openai_schema(),
        "tool_choice": "auto",
        "max_tokens": max_tokens,
    }


class OpenRouterModel:
    """Same interface as MockModel — swapping the brain changes one class and nothing else."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        prices: dict[str, float] | None = None,
        url: str = OPENROUTER_URL,
    ):
        self.name = model or os.environ.get(
            "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
        )
        # Key is held only in memory. Never write it to disk, traces, or logs.
        self._api_key = api_key if api_key is not None else os.environ.get(
            "OPENROUTER_API_KEY", ""
        )
        self._max_tokens = max_tokens
        self._prices = prices or dict(PRICE_PER_MTOK)
        self._url = url
        self.last_usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    def decide(self, system: str, turns: list[Turn]) -> Decision:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before using --live."
            )
        body = build_openrouter_request(
            model=self.name,
            system=system,
            turns=turns,
            max_tokens=self._max_tokens,
        )
        raw = self._post(body)
        self.last_usage = usage_from_response(raw, self._prices)
        return parse_openrouter_decision(raw)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # Optional OpenRouter attribution headers; no secrets.
                "HTTP-Referer": "https://github.com/cloudastructure/repo-1-agent-harness-live",
                "X-Title": "northwind-agent-harness-live",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            # Never include the request headers (they hold the key).
            raise RuntimeError(
                f"OpenRouter HTTP {e.code}: {detail}"
            ) from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenRouter network error: {e.reason}") from None


# --------------------------------------------------------------------------
# Optional: Anthropic via SDK (pip install anthropic)
# --------------------------------------------------------------------------


class AnthropicModel:
    """Drop-in replacement. Needs `pip install anthropic` and ANTHROPIC_API_KEY.

    Nothing else in the repo changes: same Decision type, same harness, same
    traces, same evals. That is the point of keeping the model behind an
    interface -- your eval suite outlives your model choice.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        import anthropic  # imported lazily so the repo runs without it
        self._client = anthropic.Anthropic()
        self.name = model
        self._max_tokens = max_tokens

    def decide(self, system: str, turns: list[Turn]) -> Decision:
        from agent import tools as tool_mod
        messages = []
        for t in turns:
            if t.role == "user":
                messages.append({"role": "user", "content": t.content})
            elif t.role == "assistant":
                messages.append({"role": "assistant", "content": t.content})
            elif t.role == "observation":
                status = "ok" if t.ok else "error"
                messages.append({
                    "role": "user",
                    "content": f"[tool:{t.tool} {status}] {t.content}",
                })

        instructions = (
            f"{system}\n\nTools available:\n{tool_mod.describe_tools()}\n\n"
            "Reply with exactly one line, either:\n"
            "  CALL <tool_name> {\"arg\": \"value\"}\n"
            "or:\n"
            "  ANSWER <your reply to the user>"
        )
        resp = self._client.messages.create(
            model=self.name,
            max_tokens=self._max_tokens,
            system=instructions,
            messages=messages or [{"role": "user", "content": "(no input)"}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if text.startswith("CALL "):
            import json
            rest = text[5:].strip()
            name, _, blob = rest.partition(" ")
            try:
                args = json.loads(blob) if blob.strip() else {}
            except json.JSONDecodeError:
                args = {}
            return ToolCall(name.strip(), args)
        return Final(text.removeprefix("ANSWER ").strip())
