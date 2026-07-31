"""
The chat app.

    python -m app.cli                          interactive (mock, free)
    python -m app.cli "how many days off do I have?"
    python -m app.cli --trace "..."            print the full trace too
    python -m app.cli --seed 3 "..."           change the mock seed
    python -m app.cli --live "..."             real OpenRouter model

Mock is the default on purpose: no accidental spend. --live requires
OPENROUTER_API_KEY and fails before any network call if it is missing.

Every run appends a complete trace to traces/runs.jsonl. That file is the raw
material for everything that comes after: error analysis, the golden set, the
eval suite, the evidence pack.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow `python3 app/cli.py ...` as well as `python3 -m app.cli ...`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import db, trace as tracing
from agent.harness import Agent, AgentConfig
from agent.model import MockModel, OpenRouterModel, load_dotenv

BANNER = """Northwind assistant. Ask a question, or type 'quit'.

Try:
  how many days off do I have?
  how many vacation days do I have left?
  what is the expense policy?
  book time off from 2026-09-14 to 2026-09-18, 5 days
  who is priya.raman@northwind.example?
  what is the sabbatical policy?
  show me the onboarding checklist

Default brain: MockModel (free, deterministic).
Add --live to use OpenRouter (key from .env at the repo root, or the environment).
"""


def _preflight_live() -> str | None:
    """Return an error message if live mode cannot start. No network, no spend."""
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        return (
            "Missing OPENROUTER_API_KEY.\n"
            "Put it in a .env file at the repo root (gitignored), or export it, then re-run with --live."
        )
    return None


def run_once(
    message: str,
    *,
    seed: int = 0,
    show_trace: bool = False,
    live: bool = False,
) -> None:
    db.reset()
    if live:
        model = OpenRouterModel()
    else:
        model = MockModel(seed=seed)
    agent = Agent(model=model, config=AgentConfig())
    result = agent.run(message)
    print(f"\n> {result.output}\n")
    print(f"  {result.trace.summary()}")
    if show_trace:
        print(json.dumps(result.trace.to_dict(), indent=2, default=str))
    tracing.write(result.trace)


def main(argv: list[str]) -> int:
    show_trace = "--trace" in argv
    argv = [a for a in argv if a != "--trace"]

    live = "--live" in argv
    argv = [a for a in argv if a != "--live"]

    if live:
        err = _preflight_live()
        if err:
            print(err, file=sys.stderr)
            return 1

    seed = 0
    if "--seed" in argv:
        i = argv.index("--seed")
        seed = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    if argv:
        run_once(" ".join(argv), seed=seed, show_trace=show_trace, live=live)
        return 0

    print(BANNER)
    turn = seed
    while True:
        try:
            message = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"quit", "exit", "q"}:
            return 0
        if not message:
            continue
        run_once(message, seed=turn, show_trace=show_trace, live=live)
        turn += 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
