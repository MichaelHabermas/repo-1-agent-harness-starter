"""
The chat app.

    python -m app.cli                          interactive
    python -m app.cli "how many days off do I have?"
    python -m app.cli --trace "..."            print the full trace too
    python -m app.cli --seed 3 "..."           change the run's seed

Every run appends a complete trace to traces/runs.jsonl. That file is the raw
material for everything that comes after: error analysis, the golden set, the
eval suite, the evidence pack.
"""

from __future__ import annotations

import json
import sys

from agent import db, trace as tracing
from agent.harness import Agent, AgentConfig
from agent.model import MockModel

BANNER = """Northwind assistant. Ask a question, or type 'quit'.

Try:
  how many days off do I have?
  how many vacation days do I have left?
  what is the expense policy?
  book time off from 2026-09-14 to 2026-09-18, 5 days
  who is priya.raman@northwind.example?
  what is the sabbatical policy?
  show me the onboarding checklist
"""


def run_once(message: str, *, seed: int = 0, show_trace: bool = False) -> None:
    db.reset()
    agent = Agent(model=MockModel(seed=seed), config=AgentConfig())
    result = agent.run(message)
    print(f"\n> {result.output}\n")
    print(f"  {result.trace.summary()}")
    if show_trace:
        print(json.dumps(result.trace.to_dict(), indent=2, default=str))
    tracing.write(result.trace)


def main(argv: list[str]) -> int:
    show_trace = "--trace" in argv
    argv = [a for a in argv if a != "--trace"]

    seed = 0
    if "--seed" in argv:
        i = argv.index("--seed")
        seed = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    if argv:
        run_once(" ".join(argv), seed=seed, show_trace=show_trace)
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
        run_once(message, seed=turn, show_trace=show_trace)
        turn += 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
