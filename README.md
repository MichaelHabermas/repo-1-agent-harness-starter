# Section 1 — The agent and its harness (Starter)

**Slides 1–11** · Northwind internal assistant · a hands-on companion to *Evals, Golden Sets & Harnesses*.

This is the starter. It runs, and it works on the happy path. That is the trap. Your job is in `TASKS.md`.

You have a working chat agent. It answers questions, calls tools, and demos beautifully.

It also has no budget, no permissions, no retries, no context policy, and a trace that records almost nothing. When it goes wrong — and you'll make it go wrong in the first five minutes — you cannot say why, cannot reproduce it, and cannot prove it to anyone.

By the end you'll have built the five harness layers, and the same agent will be one you could defend in a review.

## Getting started

Default path: no dependencies, no API key, no network.

```bash
python3 --version          # 3.10 or newer
python -m app.cli          # talk to it
make help                  # everything you can run
```

The model is a deterministic stand-in for a hosted LLM, in `agent/model.py`. That is a feature, not a shortcut: it runs offline in milliseconds, and its failures are planted and reproducible, so everyone in the room finds the same bugs instead of a different sample of model noise. Swapping in a real model is one class — `AnthropicModel` is at the bottom of the same file, and nothing else in the repo changes.

## What's in here

```
agent/
  db.py           the company's data: employees, PTO, handbook
  tools.py        the five tools, their schemas, and their effects
  model.py        the decision layer (swap for a real model here)
  context.py      what the model sees, and what gets compacted away
  guardrails.py   budgets, permissions, preconditions, egress
  harness.py      the loop  <- your work happens here
  trace.py        the record of what happened
app/cli.py        the chat app
tests/            unit tests for the harness
```

## Break it first

Before you write anything, spend five minutes making it misbehave:

```bash
python3 -m app.cli "who is nobody"                        # unhandled exception
python3 -m app.cli "what is the expense policy?"          # count the tool calls
python3 -m app.cli "what is the sabbatical policy?"       # there isn't one
python3 -m app.cli --trace "how many days off do I have?" # what's missing?
```

Everything you find is a harness problem, not a model problem. That's the point.

## The product

Northwind's internal assistant answers employee questions about HR and IT: the handbook, the directory, and paid time off. It has five tools, one of which changes company state. It is exactly the kind of thing a team ships in a fortnight and then supports for three years.

It works on the happy path. Ask it how many days off you have and it tells you.
That is the whole problem: the happy path is not the job.

---

*The mock model, the company data and the incidents in the golden set are fictional. The failure modes are not.*

## The live version

Everything above runs offline against the scripted MockModel. Same harness, real brain:

    export OPENROUTER_API_KEY=...        # openrouter.ai
    python3 app/cli.py --live "how many days off do I have?"
    python3 demo_variance.py             # one mock run vs three live runs, side by side

Live runs use a cheap model on purpose (default `google/gemini-2.5-flash-lite`,
override with `OPENROUTER_MODEL`). Cheap models make real mistakes — invented
tool names, malformed arguments, skipped steps — which is exactly what the
validation, retry, and budget layers are for. Cost is capped at $0.05 per run;
a typical run is well under a cent. No key set → `--live` says so and exits.

One thing worth seeing once: the TODOs below aren't implemented yet, so live
mode here runs without its guardrails. Watch what that looks like, then build them.
